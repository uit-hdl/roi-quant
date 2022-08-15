import os
import shutil
import datetime  
import natsort

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def rm_n_mkdir(dir_path):
    '''
    Remove (if was present) and create new directory.

    Parameters:
    - dir_path (str):       full path of the directory

    Returns: None
    '''
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class PatchDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.imgs = natsort.natsorted(os.listdir(self.root))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_name).convert("RGB")
        tensor_img = self.transform(img)
        return (tensor_img)

tf = {
    'unify_to_tensor': transforms.Compose([
                            transforms.Resize((512, 512)),
                            transforms.ToTensor()]),
    'to_image': transforms.ToPILImage()
}

if __name__ == "__main__":
    MODEL = 'mobilenetv3.pt'
    DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    BATCH_SIZE = 256
    IMG_DIR = 'input_dir_with_images/'
    SORTED_DIR = 'qc_output_dir_for_next_step/test/Images'

    patch_dataset = PatchDataset(IMG_DIR, transform=tf['unify_to_tensor'])
    patch_dataloader = DataLoader(patch_dataset, batch_size=BATCH_SIZE, shuffle=False)
    imgs = patch_dataloader.dataset.imgs

    model = torch.load(f'patch_class/models/{MODEL}')
    model = model.to(DEVICE)
    model.eval()

    labels = []
    with torch.no_grad():
        for batch_idx, (data) in enumerate(patch_dataloader):
            print(f"{datetime.datetime.now()} - batch {batch_idx+1}")
            data = data.to(DEVICE)
            output = model(data)
            (_, predictions) = output.max(1)
            labels.extend(predictions.cpu().numpy())
    
    table = {'Images': imgs, 'Labels': labels} 
    result_sorted = pd.DataFrame(table).query("Labels == 2")
    # filter label 0,1 (necrotic, lung), leave 2 (stroma + tils)
    rm_n_mkdir(SORTED_DIR)

    for img_name in list(result_sorted['Images']):
        shutil.copy2(os.path.join(IMG_DIR, img_name), 
            os.path.join(SORTED_DIR, img_name))

    



    
    



