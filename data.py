import numpy as np
import pandas as pd
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split
import cv2


def load_classes(classes_json):
    with open(classes_json, 'r') as f:
        classes = json.load(f)
    print("Number of classes:", len(classes))
    return classes

def create_df(image_path):
    name = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            name.append(filename.split('.')[0])
    df = pd.DataFrame({'id': name}, index=np.arange(0, len(name)))
    print('Total Images: ', len(df))

    return df


def split_dataset(df, test_size=0.1, val_size=0.15, random_state=19):
    X_trainval, X_test = train_test_split(df['id'].values, test_size=test_size, random_state=random_state)
    X_train, X_val = train_test_split(X_trainval, test_size=val_size, random_state=random_state)
    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))
    print('Test Size    : ', len(X_test))

    return X_train, X_val, X_test


def display_images_with_mask(df, IMAGE_PATH, LABEL_PATH, num_images=2):
    # Create subplots for the images
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 6*num_images))

    for i in range(num_images):
        # Load the original image and the mask
        img_name = df['id'][i]
        img = Image.open(IMAGE_PATH +"/"+ img_name + '.tif')
        mask = Image.open(LABEL_PATH +"/"+ img_name + '.tif')

        # Display original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')

        # Display mask applied on the image
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask, alpha=0.9)
        axes[i, 1].set_title('Picture with Mask Applied')

        # Print the name of the image above the subplots
        axes[i, 0].set_xlabel(img_name)
        axes[i, 1].set_xlabel(img_name)

    plt.tight_layout()
    plt.show()


def display_augmented_images_with_mask(df, IMAGE_PATH, LABEL_PATH, t_train, num_images=2):
    # Create subplots for the augmented images
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 6*num_images))

    for i in range(num_images):
        # Load the original image and the mask
        img_name = df['id'][i]
        img = Image.open(IMAGE_PATH +"/"+ img_name + '.tif')
        mask = Image.open(LABEL_PATH +"/"+ img_name + '.tif')

        # Apply augmentation to the image
        augmented_img = t_train(image=np.array(img))['image']
        augmented_mask = t_train(image=np.array(mask))['image']

        # Display augmented image
        axes[i, 0].imshow(augmented_img)
        axes[i, 0].set_title('Augmented Image')

        # Display mask applied on the augmented image
        axes[i, 1].imshow(augmented_img)
        axes[i, 1].imshow(augmented_mask, alpha=0.9)
        axes[i, 1].set_title('Picture with Mask Applied')

        # Print the name of the image above the subplots
        axes[i, 0].set_xlabel(img_name)
        axes[i, 1].set_xlabel(img_name)

    plt.tight_layout()
    plt.show()
