import albumentations as A
import cv2


def create_augmentations():
    t_train = A.Compose([
        A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
        A.GaussNoise()])

    t_val = A.Compose([
        A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.GridDistortion(p=0.2)])

    return t_train, t_val
