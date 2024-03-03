import albumentations as A
import cv2
from TestDataset import TestDataset


def creating_test_set(image_path, label_path, X_test):
    t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    test_set = TestDataset(image_path, label_path, X_test, transform=t_test)

    return test_set
