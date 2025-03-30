import os
import csv

from PIL import Image, ImageOps
import random

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

from scipy.ndimage import gaussian_filter



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




csvFile = '@CLaMM-filelist.csv'
folder = '../clamm/2016_training/CLaMM_Training_Data_Set'
train_2016 = build_dataset(folder + "/" + csvFile, folder)
dataset = train_2016


folder = '../clamm/2017_training/ICDAR2017_CLaMM_Training'
csvFile = '@ICDAR2017_CLaMM_Training.csv'
new_ds = build_dataset(folder + "/" + csvFile, folder)
dataset.update(new_ds)


csvFile = '@CLaMM_task1.csv'
folder = '../clamm/2016_task1/CLaMM_task1'
new_ds = build_dataset(folder + "/" + csvFile, folder)
dataset.update(new_ds)


folder = '../clamm/2016_task2/CLaMM_task2'
csvFile = '@CLaMM_task2.csv'
new_ds = build_dataset(folder + "/" + csvFile, folder)
dataset.update(new_ds)


folder = '../clamm/2017_task1_task3'
csvFile = '@ICDAR2017_CLaMM_task1_task3.csv'
new_ds = build_dataset(folder + "/" + csvFile, folder, 1)
dataset.update(new_ds)


folder = '../clamm/2017_task2_task4'
csvFile = '@ICDAR2017_CLaMM_task2_task4.csv'
new_ds = build_dataset(folder + "/" + csvFile, folder, 1)
dataset.update(new_ds)
# dataset = new_ds

print(len(dataset), 'total images')




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






def crop_sample(image):
    crop_dim = 300
    
    img_width, img_height = image.size
    
    margin_x = int(img_width * 0.05)
    margin_y = int(img_height * 0.05)
    
    max_x = img_width - crop_dim - margin_x
    max_y = img_height - crop_dim - margin_y
    
    left = random.randint(margin_x, max_x)
    upper = random.randint(margin_y, max_y)
    
    crop_box = (left, upper, left + crop_dim, upper + crop_dim)
    cropped_image = image.crop(crop_box)
    return cropped_image