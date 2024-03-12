import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy

def load_images(folder_path, gans=False):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    images = []
    if gans: 
        s = f'{folder_path}/*.png'
    else:
        s = f'{folder_path}/*.jpeg'
    for filename in glob.glob(s): # Adjust the path and file type if necessary
        with open(filename, 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = transform(image).unsqueeze(0)
            images.append(image)
    return torch.cat(images[:-1], dim=0)

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    assert imgs.size(2) == 299 and imgs.size(3) == 299, 'Inception v3 requires 299x299 input'

    # Set up the model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    if cuda:
        inception_model.cuda()

    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Compute the predictions for the images
    preds = np.zeros((imgs.size(0), 1000))

    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i + batch_size]
        if cuda:
            batch = batch.cuda()
        batch_preds = get_pred(batch)
        preds[i:i + batch_size] = batch_preds

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (preds.shape[0] // splits): (k+1) * (preds.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# Load your images
images = load_images('./model_outptus/diffusion', gans=False)  # Update this with your folder path

# Calculate the Inception Score
mean, std = inception_score(images, cuda=False, batch_size=32, splits=10)
print(f"Inception Score for Diffusion: Mean = {mean}, Std = {std}")

# Load your images
images = load_images('./model_outptus/gan', gans=True)  # Update this with your folder path

# Calculate the Inception Score
mean, std = inception_score(images, cuda=False, batch_size=32, splits=10)
print(f"Inception Score for Gan: Mean = {mean}, Std = {std}")
