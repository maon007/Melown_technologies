## Import Libraries
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import cv2


class SegmentationDataset(Dataset):
    """
    This class is designed to load images and corresponding masks from specified paths and apply transformations to them if needed. 
    """

    def __init__(self, img_path, label_path, rendered_label_path, X, mean, std, transform=None, patch=False,
                 use_rendered_labels=False):
        self.img_path = img_path
        self.label_path = label_path
        self.rendered_label_path = rendered_label_path  # New addition
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        self.use_rendered_labels = use_rendered_labels  # New addition

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        print(f"Image path: {self.img_path + '/' + str(self.X[idx]) + '.tif'}")
        img = cv2.imread(self.img_path + "/" + self.X[idx] + '.tif')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask or rendered label based on the flag
        if self.use_rendered_labels:
            mask = cv2.imread(self.rendered_label_path + "/" + self.X[idx] + '.tif',
                              cv2.IMREAD_GRAYSCALE)  # Assuming rendered labels are in PNG format
        else:
            mask = cv2.imread(self.label_path + "/" + self.X[idx] + '.tif', cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask):

        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        return img_patches, mask_patches