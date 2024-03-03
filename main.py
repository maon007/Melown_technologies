import torch
import torch.nn as nn
from SegmentationDataset import SegmentationDataset
from data import create_df, split_dataset
from augumentation import create_augmentations
from evaluation import predict_image_mask_miou, miou_score, pixel_acc
from model import create_model
from visualization import visualize_history, visualize_prediction
from train import fit
from test import creating_test_set
from dataloader import get_dataloaders
from config import IMAGE_PATH, CLASSES_JSON, LABEL_PATH, RENDERED_LABEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentationModel:
    def __init__(self, image_path, label_path, rendered_label_path, classes_json):
        self.image_path = image_path
        self.classes_json = classes_json
        self.label_path = label_path
        self.rendered_label_path = rendered_label_path
        self.model = None  # Initialize model as None
        self.X_test = None  # Initialize X_test as None

    def train(self, epochs=3):
        # Load data
        df = create_df(self.image_path)

        X_train, X_val, X_test = split_dataset(df)
        self.X_test = X_test  # Store X_test for later evaluation

        # Load augmentations
        t_train, t_val = create_augmentations()

        # Create datasets
        train_dataset = SegmentationDataset(img_path=self.image_path, label_path=self.label_path,
                                            rendered_label_path=self.rendered_label_path,
                                            X=X_train, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                            transform=t_train, patch=False,
                                            use_rendered_labels=False)

        val_dataset = SegmentationDataset(img_path=self.image_path, label_path=self.label_path,
                                          rendered_label_path=self.rendered_label_path,
                                          X=X_val, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                          transform=t_val, patch=False,
                                          use_rendered_labels=True)

        train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

        # Create model and move it to the appropriate device
        self.model = create_model(encoder_name='mobilenet_v2', encoder_weights='imagenet', encoder_depth=3,
                                  decoder_channels=[128, 64, 32]).to(device)

        # Define optimizers and schedulers
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                        epochs=epochs,
                                                        steps_per_epoch=len(train_loader))

        # Training loop
        history = fit(epochs, self.model, train_loader, val_loader, criterion=nn.CrossEntropyLoss(),
                      optimizer=optimizer, scheduler=scheduler, device=device)

        # visualize_history(history)
        # Saving the model (consider saving more than just the model for comprehensive loading later)
        torch.save(self.model.state_dict(), 'Unet-Mobilenet.pth')

        return history

    def evaluate(self):
        if self.X_test is None or self.model is None:
            print("Model or X_test not available. Please train the model first.")
            return

        test_set = creating_test_set(image_path=self.image_path, label_path=self.label_path, X_test=self.X_test)
        self.model.eval()  # Set the model to evaluation mode

        # Move model to the appropriate device
        self.model.to(device)

        # Results
        mob_miou = miou_score(self.model, test_set, device)
        mob_acc = pixel_acc(self.model, test_set, device)

        image, mask = test_set[0]  # Assuming test_set is ready for device transfer
        image, mask = image.to(device), mask.to(device)
        pred_mask, score = predict_image_mask_miou(self.model, image, mask, device)

        pred = visualize_prediction(image.cpu(), mask.cpu(), pred_mask.cpu(),
                                    score)  # Move tensors back to CPU for visualization
        return pred


if __name__ == "__main__":
    # Instantiate SegmentationModel
    model = SegmentationModel(image_path=IMAGE_PATH, label_path=LABEL_PATH, rendered_label_path=RENDERED_LABEL_PATH,
                              classes_json=CLASSES_JSON)

    # Train the model
    history = model.train(epochs=12)  # Assuming a combined total of pre-training and fine-tuning epochs

    # Evaluate the model
    prediction = model.evaluate()
