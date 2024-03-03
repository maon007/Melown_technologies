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
        self.label_path = label_path
        self.rendered_label_path = rendered_label_path
        self.classes_json = classes_json
        self.model = None
        self.X_test = None

    def train(self):
        # Data loading and splitting
        df = create_df(self.image_path)
        X_train, X_val, X_test = split_dataset(df)
        self.X_test = X_test

        # Augmentations
        t_train, t_val = create_augmentations()

        # Datasets
        train_dataset = SegmentationDataset(img_path=self.image_path, label_path=self.label_path,
                                            rendered_label_path=self.rendered_label_path,
                                            X=X_train, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                            transform=t_train, use_rendered_labels=False)

        val_dataset = SegmentationDataset(img_path=self.image_path, label_path=self.label_path,
                                          rendered_label_path=self.rendered_label_path,
                                          X=X_val, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                          transform=t_val, use_rendered_labels=True)

        # DataLoaders
        train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

        # Model
        self.model = create_model(encoder_name='mobilenet_v2', encoder_weights='imagenet', encoder_depth=3,
                                  decoder_channels=[128, 64, 32]).to(device)

        # Pre-training
        pre_training_epochs = 4
        fine_tuning_epochs = 8
        criterion = nn.CrossEntropyLoss()

        # Pre-training Phase
        pre_training_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        pre_training_scheduler = torch.optim.lr_scheduler.OneCycleLR(pre_training_optimizer, max_lr=1e-3,
                                                                     epochs=pre_training_epochs,
                                                                     steps_per_epoch=len(train_loader))

        print("** Pre-training Stage **")
        pre_training_history = fit(epochs=pre_training_epochs, model=self.model, train_loader=train_loader,
                                   val_loader=val_loader, criterion=criterion, optimizer=pre_training_optimizer,
                                   scheduler=pre_training_scheduler, device=device)

        # Fine-tuning Phase
        fine_tuning_optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        fine_tuning_scheduler = torch.optim.lr_scheduler.OneCycleLR(fine_tuning_optimizer, max_lr=1e-4,
                                                                    epochs=fine_tuning_epochs,
                                                                    steps_per_epoch=len(train_loader))

        print("** Fine-tuning Stage **")
        fine_tuning_history = fit(epochs=fine_tuning_epochs, model=self.model, train_loader=train_loader,
                                  val_loader=val_loader, criterion=criterion, optimizer=fine_tuning_optimizer,
                                  scheduler=fine_tuning_scheduler, device=device)

        torch.save(self.model.state_dict(), 'Unet-Mobilenet-FineTuned.pth')

        return pre_training_history, fine_tuning_history

    def evaluate(self):
        if self.X_test is None or self.model is None:
            print("Model or X_test not available. Please train the model first.")
            return

        test_set = creating_test_set(image_path=self.image_path, label_path=self.label_path, X_test=self.X_test)
        self.model.eval()
        self.model.to(device)

        mob_miou = miou_score(self.model, test_set, device)
        mob_acc = pixel_acc(self.model, test_set, device)

        image, mask = test_set[0]
        image, mask = image.to(device), mask.to(device)
        pred_mask, score = predict_image_mask_miou(self.model, image, mask, device)

        pred = visualize_prediction(image.cpu(), mask.cpu(), pred_mask.cpu(), score)
        return pred

if __name__ == "__main__":
    model = SegmentationModel(image_path=IMAGE_PATH, label_path=LABEL_PATH,
                              rendered_label_path=RENDERED_LABEL_PATH, classes_json=CLASSES_JSON)
    history = model.train()
    prediction = model.evaluate()
