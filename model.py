import segmentation_models_pytorch as smp
from data import load_classes
from config import CLASSES_JSON


def create_model(encoder_name, encoder_weights, encoder_depth, decoder_channels):
    num_classes = len(load_classes(classes_json=CLASSES_JSON))

    # Define the U-Net model architecture using segmentation_models_pytorch
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels  # Customize decoder channels as needed
    )

    return model
