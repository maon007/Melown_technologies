from torch.utils.data import DataLoader


def get_dataloaders(original_label_dataset, rendered_label_dataset, batch_size=3):
    original_label_dataloader = DataLoader(original_label_dataset, batch_size=3, shuffle=True)
    rendered_label_dataloader = DataLoader(rendered_label_dataset, batch_size=3, shuffle=True)
    return original_label_dataloader, rendered_label_dataloader
