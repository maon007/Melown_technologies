## Import Libraries
import torch
from torch.utils.data import Dataset

from PIL import Image
import cv2


class TestDataset(Dataset):

    def __init__(self, img_path, label_path, X, transform=None):
        self.img_path = img_path
        self.label_path = label_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        print("Index:", idx)
        print("Length of X:", len(self.X))
        img = cv2.imread(self.img_path + "/" + self.X[idx] + '.tif')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.label_path + "/" + self.X[idx] + '.tif', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        mask = torch.from_numpy(mask).long()

        return img, mask