import matplotlib.pyplot as plt
import seaborn as sns

def visualize_history(history):
    sns.set_style("darkgrid")  # Set a visually appealing style

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Loss plot
    sns.lineplot(x=range(len(history['train_loss'])), y='train_loss', data=history, label='Train Loss', ax=axs[0])
    sns.lineplot(x=range(len(history['val_loss'])), y='val_loss', data=history, label='Val Loss', ax=axs[0])
    axs[0].set_title('Loss per Epoch')
    axs[0].set_ylabel('Loss')

    # Mean IoU plot
    sns.lineplot(x=range(len(history['train_miou'])), y='train_miou', data=history, label='Train mIoU', ax=axs[1])
    sns.lineplot(x=range(len(history['val_miou'])), y='val_miou', data=history, label='Val mIoU', ax=axs[1])
    axs[1].set_title('Mean IoU per Epoch')
    axs[1].set_ylabel('Mean IoU')

    # Accuracy plot
    sns.lineplot(x=range(len(history['train_acc'])), y='train_acc', data=history, label='Train Accuracy', ax=axs[2])
    sns.lineplot(x=range(len(history['val_acc'])), y='val_acc', data=history, label='Val Accuracy', ax=axs[2])
    axs[2].set_title('Accuracy per Epoch')
    axs[2].set_ylabel('Accuracy')

    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()


def visualize_prediction(image, mask, predicted_mask, score):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(image)
    ax1.set_title('Picture');

    ax2.imshow(mask)
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(predicted_mask)
    ax3.set_title('UNet-MobileNet | mIoU {:.3f}'.format(score))
    ax3.set_axis_off()
