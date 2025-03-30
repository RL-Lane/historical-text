import csv

from PIL import Image, ImageOps
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models

import numpy as np

from scipy.ndimage import gaussian_filter

from blf_torch import BilateralFilter, DEVICE






# print('check for train dataset rows:')
with open('train_dataset.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)  # Reads the CSV into a list of dictionaries
    train = [{**row, 'label_num': int(row['label_num'])} for row in reader]  # Convert label_num to int
# Print the dictionary
# for row in train[:3]:
#     print(row)

print('loading image metadata...', end='')
# print('check for test dataset rows:')
with open('test_dataset.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)  # Reads the CSV into a list of dictionaries
    test = [{**row, 'label_num': int(row['label_num'])} for row in reader]  # Convert label_num to int
# Print the dictionary
# for row in test[:3]:
#     print(row)



# These datasets were pre-shuffled, so we should be able to take a small sample for testing out algorithms.
mini_length = 100
train_mini = train[:mini_length]
# print('\ntrain_mini length:',len(train_mini))

print(' done')











# https://clamm.irht.cnrs.fr/icdar-2017/data-set/

script_conversion = {
    '1':"Caroline",
    '2':"Cursiva",
    '3':"Half Uncial",
    '4':"Humanistic",
    '5':"Humanistic Cursive",
    '6':"Hybrida",
    '7':"Praegothica",
    '8':"Semihybrida",
    '9':"Semitextualis",
    '10':"Southern Textualis",
    '11':"Textualis",
    '12':"Uncial"
}

reverse_script_conversion = {v: int(k) for k, v in script_conversion.items()}

def convert_to_script(label_number):
    """
    Convert a label number (e.g., '11') to its script name.
    """
    return script_conversion.get(str(label_number), "Unknown")

def clean_label(label):
    # Replace '_' with ' ', convert to title case, and strip spaces
    return label.replace("_", " ").title().strip()

def convert_to_number(script_name):
    """
    Convert a script name (e.g., 'Textualis') to its corresponding number.
    """
    cleaned_script_name = clean_label(script_name)
    return reverse_script_conversion.get(cleaned_script_name, -1)

def build_dataset(csv_path, image_folder, icdar = 0):


        # Routine cleaning function

    
    # Step 1: Read the CSV and store labels in a dictionary
    label_dict = {}
    with open(csv_path, mode='r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Assuming the first row is header, skip it if needed
        next(reader)
        for row in reader:
            filename = row[0+icdar]  # Adjust index based on your CSV structure
            label = row[1+icdar]  # Adjust index based on your CSV structure
            label = str(label)
            label = clean_label(script_conversion.get(label, label))
            if (label in script_conversion):
                label = script_conversion[label]
            label_dict[filename] = label

    # Step 2: Match images to labels and store the full path
    dataset = {}
    with os.scandir(image_folder) as entries:
        for entry in entries:
            if entry.is_file():
                image_id = entry.name
                
                if image_id in label_dict:  # Only add if we have a matching label
                    dataset[image_id] = {
                        'filepath': os.path.join(image_folder, entry.name),
                        'label': label_dict[image_id],
                        'label_num': reverse_script_conversion[label_dict[image_id]]
                    }

    print(f"Dataset created with {len(dataset)} items.")
    return dataset















class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        is_pillow = isinstance(image, Image.Image)
        if is_pillow:
            image = TF.to_tensor(image)
        noise = torch.randn_like(image) * self.std + self.mean
        noisy_image = torch.clamp(image + noise, 0, 1)
        if is_pillow:
            return TF.to_pil_image(noisy_image)
        return noisy_image


class AddSpeckleNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        is_pillow = isinstance(image, Image.Image)
        if is_pillow:
            image = TF.to_tensor(image)
        noise = torch.randn_like(image) * self.std + self.mean
        noisy_image = torch.clamp(image + image * noise, 0, 1)
        if is_pillow:
            return TF.to_pil_image(noisy_image)
        return noisy_image


class AddSaltAndPepperNoise:
    def __init__(self, amount=0.02, salt_vs_pepper=0.5):
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, image):
        is_pillow = isinstance(image, Image.Image)
        if is_pillow:
            image = TF.to_tensor(image)
        noisy_image = image.clone()
        num_pixels = int(self.amount * image.numel())
        salt_indices = torch.randint(0, image.numel(), (int(num_pixels * self.salt_vs_pepper),), device=image.device)
        pepper_indices = torch.randint(0, image.numel(), (int(num_pixels * (1 - self.salt_vs_pepper)),), device=image.device)
        noisy_image.view(-1)[salt_indices] = 1.0
        noisy_image.view(-1)[pepper_indices] = 0.0
        if is_pillow:
            return TF.to_pil_image(noisy_image)
        return noisy_image

# Prepare a bilateral filter class for the pipeline.
# blf() above uses CPU and is too slow.  Below uses CUDA if available (DEVICE).



class ApplyBilateralFilter:
    def __init__(self, kernel_size=5, sigma_space=5, sigma_color=0.1):
        self.kernel_size = kernel_size
        self.sigma_space = sigma_space
        self.sigma_color = sigma_color

    def __call__(self, img):
        # Convert PIL image to tensor
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)#.to(DEVICE)

        # Initialize the bilateral filter (dynamic size detection)
        filter_device = torch.device("cuda:0")   # For BilateralFilter
        bilateral_filter = BilateralFilter(
            channels=img_tensor.shape[1],
            k=self.kernel_size,
            height=img_tensor.shape[2],
            width=img_tensor.shape[3],
            sigma_space=self.sigma_space,
            sigma_color=self.sigma_color,
            device=filter_device
        )

        # Apply the filter
        with torch.no_grad():
            filtered_tensor = bilateral_filter(img_tensor)

        # Convert the tensor back to a PIL image
        filtered_img = transforms.ToPILImage()(filtered_tensor.squeeze(0).to(DEVICE))
        return filtered_img
        # return filtered_tensor


def repeat_channels(x):
    return x.repeat(3, 1, 1)

# geometric transform
transform_pipeline = transforms.Compose([
    # geometric
    transforms.RandomApply(      [transforms.RandomAffine(degrees=0, shear=10, interpolation=Image.BICUBIC)              ], p=0.5)
    ,transforms.RandomApply(     [transforms.RandomRotation(degrees=15, interpolation=Image.BICUBIC)                     ], p=0.5)
    ,transforms.RandomApply(     [transforms.RandomPerspective(distortion_scale=0.2, p=0.3, interpolation=Image.BICUBIC) ], p=0.2)
    ,transforms.RandomResizedCrop(
        size=(670,670),
        scale=(0.8, 1.2),
        ratio=(0.8, 1.2),
        interpolation=Image.BICUBIC
    )
    # ,ApplyBilateralFilter(kernel_size=5, sigma_space=5, sigma_color=0.1)

    # color / photo effects
    ,transforms.RandomApply(     [transforms.GaussianBlur(kernel_size=(3,3))                                             ], p=0.2)

    ,transforms.RandomApply([transforms.RandomChoice([
                            AddGaussianNoise(mean=0.0, std=0.1),
                            AddSpeckleNoise(mean=0.0, std=0.1),
                            AddSaltAndPepperNoise(amount=0.02, salt_vs_pepper=0.5)
                        ])                                                                                               ], p = 0.25)


    
    ,transforms.CenterCrop((500,500)) # ResNet50 expects 224x224
    ,transforms.ToTensor()
    # ,transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel grayscale to 3-channel
    ,transforms.Lambda(repeat_channels)
    ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.500, 0.225])  # Standard normalization for ResNet50
])

test_transform_pipeline = transforms.Compose([
    # geometric
    ApplyBilateralFilter(kernel_size=5, sigma_space=5, sigma_color=0.1)

    
    ,transforms.CenterCrop((500,500)) # ResNet50 expects 224x224
    ,transforms.ToTensor()
    # ,transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel grayscale to 3-channel
    ,transforms.Lambda(repeat_channels)
    ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet50
])



minimal_preprocessing_pipeline = transforms.Compose([
    # geometric

    transforms.CenterCrop((500,500)) # ResNet50 expects 224x224
    ,transforms.ToTensor()
    # ,transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel grayscale to 3-channel
    ,transforms.Lambda(repeat_channels)
    ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet50
])
















class ScriptDataset(Dataset):
    def __init__(self, dataset, transform=None, multiplier=10, max_size=None):
        """
        dataset: List of dicts, each containing 'filepath', 'label', and 'label_num'
        transform: torchvision transforms (augmentations + preprocessing)
        multiplier: Number of times each raw image is virtually repeated
        max_size: Upper limit on the dataset size (optional)
        """
        self.dataset = dataset  # Now a list of dicts, not a dict itself
        self.transform = transform
        self.multiplier = multiplier
        
        # Compute virtual dataset size
        self.virtual_size = len(self.dataset) * self.multiplier

        # Apply max_size limit if provided
        if max_size is not None:
            self.virtual_size = min(self.virtual_size, max_size)

    def crop_sample(self, image, crop_dim = 670):
        """Crop a random 300x300 region."""
        img_width, img_height = image.size
        margin_x = int(img_width * 0.05)
        margin_y = int(img_height * 0.05)
        max_x = img_width - crop_dim - margin_x
        max_y = img_height - crop_dim - margin_y

        left = random.randint(margin_x, max_x)
        upper = random.randint(margin_y, max_y)
        crop_box = (left, upper, left + crop_dim, upper + crop_dim)
        return image.crop(crop_box)

    def __len__(self):
        """Return the virtual dataset size (capped at max_size if set)."""
        return self.virtual_size

    def __getitem__(self, idx):
        # Map virtual index back to the original dataset
        real_idx = idx % len(self.dataset)
        row = self.dataset[real_idx]

        # Load image
        image = Image.open(row["filepath"]).convert("L")

        # Apply random crop
        image = self.crop_sample(image)

        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)
            
        #  Python counts from 0, but the dataset counts from 1.  We will have to account for this later
        corrected_label = row['label_num'] - 1
        # Return image and numerical label as a tuple
        return image, corrected_label