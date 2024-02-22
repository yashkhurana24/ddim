import argparse
import torch
from model import UNet
from scheduler import DDIMScheduler
from utils import _grayscale_to_rgb, save_images, normalize_to_neg_one_to_one
from dataset import DiffusionDataset
from torchinfo import summary
from model_ema import EMA
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from torchvision import utils

from diffusers.optimization import get_scheduler
from tqdm import tqdm
from torch.nn import functional as F

from torchvision import transforms
from PIL import Image
import os
from datetime import datetime

# n_timesteps = 1000
# n_inference_timesteps = 250


def main(args):
        
    if args.mode not in ["train", "eval", "inference"]:
        raise ValueError(f"Invalid mode {args.mode}. Available modes are train, eval, inference.")
    
    device = torch.device(args.device)
    n_timesteps = args.n_train_timesteps
    n_inference_timesteps = args.n_inference_timesteps

    if args.mode == "train":

        model = UNet(3, image_size=args.resolution, hidden_dims=[64, 128, 256, 512])
        noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")
        if args.pretrained_model_path:
            pretrained = torch.load(args.pretrained_model_path)["model_state"]
            model.load_state_dict(pretrained)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        )
        
        tfms = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            transforms.Lambda(_grayscale_to_rgb),
            transforms.ToTensor()
        ])

        dataset = DiffusionDataset(args.dataset_path, split='train', transform=tfms)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
        steps_per_epcoch = len(train_dataloader)

        total_num_steps = (steps_per_epcoch * args.num_epochs) // args.gradient_accumulation_steps
        total_num_steps += int(total_num_steps * 10/100)
        gamma = args.gamma
        ema = EMA(model, gamma, total_num_steps)

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=total_num_steps,
        )
        
        summary(model, [(1, 3, args.resolution, args.resolution), (1,)], verbose=1)

        scaler = GradScaler(enabled=False)
        # scaler = GradScaler(enabled=args.fp16_precision)
        
        global_step = 0
        losses = []
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(total=steps_per_epcoch)
            progress_bar.set_description(f"Epoch {epoch}")
            losses_log = 0
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["image"].to(device)
                clean_images = normalize_to_neg_one_to_one(clean_images)

                batch_size = clean_images.shape[0]
                noise = torch.randn(clean_images.shape).to(device)
                timesteps = torch.randint(0,
                                        noise_scheduler.num_train_timesteps,
                                        (batch_size,),
                                        device=device).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                        timesteps)

                optimizer.zero_grad()
                with autocast(enabled=args.fp16_precision):
                    noise_pred = model(noisy_images, timesteps)["sample"]
                    loss = F.l1_loss(noise_pred, noise)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                ema.update_params(gamma)
                gamma = ema.update_gamma(global_step)

                if args.use_clip_grad:
                    clip_grad_norm_(model.parameters(), 1.0)

                lr_scheduler.step()

                progress_bar.update(1)
                losses_log += loss.detach().item()
                logs = {
                    "loss_avg": losses_log / (step + 1),
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "gamma": gamma
                }

                progress_bar.set_postfix(**logs)
                global_step += 1

                # Generate sample images for visual inspection
                if global_step % args.save_model_steps == 0:
                    ema.ema_model.eval()
                    with torch.no_grad():
                        # has to be instantiated every time, because of reproducibility
                        generator = torch.manual_seed(0)
                        generated_images = noise_scheduler.generate(
                            ema.ema_model,
                            num_inference_steps=n_inference_timesteps,
                            generator=generator,
                            eta=1.0,
                            use_clipped_model_output=True,
                            batch_size=args.eval_batch_size,
                            output_type="numpy")

                        print(f'Saving images for step {global_step}')

                        save_images(generated_images, epoch, args)

                        torch.save(
                            {
                                'model_state': model.state_dict(),
                                'ema_model_state': ema.ema_model.state_dict(),
                                'optimizer_state': optimizer.state_dict(),
                            }, args.output_dir)

            progress_bar.close()
            losses.append(losses_log / (step + 1))

    elif args.mode == "inference":
        if args.pretrained_model_path is None:
            raise ValueError("Pretrained model path is required for inference mode.")
        
        model = UNet(3, image_size=args.resolution, hidden_dims=[64, 128, 256, 512])
        noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

        # if device == "cpu":
        #     pretrained = torch.load(args.pretrained_model_path, map_location=device)["model_state"]
        # else:
        #     pretrained = torch.load(args.pretrained_model_path)["model_state"]
        pretrained = torch.load(args.pretrained_model_path, map_location=device)["model_state"]
        model.load_state_dict(pretrained)
        model = model.to(device)

        with torch.no_grad():
            # has to be instantiated every time, because of reproducibility
            generator = torch.manual_seed(0)
            generated_images = noise_scheduler.generate(
                model,
                num_inference_steps=n_inference_timesteps,
                generator=generator,
                eta=0.5,
                use_clipped_model_output=True,
                batch_size=args.eval_batch_size,
                output_type="numpy")

            images = generated_images["sample"]
            images_processed = (images * 255).round().astype("uint8")

            current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
            out_dir = f"./{args.samples_dir}/{current_date}/"
            os.makedirs(out_dir)
            for idx, image in enumerate(images_processed):
                image = Image.fromarray(image)
                image.save(f"{out_dir}/{idx}.jpeg")

            utils.save_image(generated_images["sample_pt"],
                            f"{out_dir}/grid.jpeg",
                            nrow=args.eval_batch_size // 4)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Main script for training and inference.")
    
    
    # starters
    parser.add_argument("--mode", type=str, default="train",
                        help="train/eval/inference")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="cuda, cpu")
    parser.add_argument('--dataset_path',
                        type=str,
                        default='../stanfordCars',
                        help='Path to dataset')
    parser.add_argument("--logging_dir", 
                        type=str, 
                        default="logs")
    parser.add_argument("--pretrained_model_path",
                    type=str,
                    default=None,
                    help="Path to pretrained model")

    # output
    parser.add_argument("--samples_dir", type=str, default="samples")
    parser.add_argument("--dataset_name", type=str, default="stanfordcars")
    parser.add_argument("--output_dir", type=str, default="trained_models/ddpm-model-1.pth")

    # training parameters
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_model_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument('--fp16_precision',
                        action='store_true',
                        help='Whether to use 16-bit precision for GPU training')
    parser.add_argument('--gamma',
                    default=0.996,
                    type=float,
                    help='Initial EMA coefficient')

    parser.add_argument('--n_train_timesteps',
                        default=1000,
                        type=int,
                        help='Number of training steps')
    parser.add_argument('--n_inference_timesteps',
                        default=100,
                        type=int,
                        help='Number of inference steps')


    args = parser.parse_args()

    
    main(args)
