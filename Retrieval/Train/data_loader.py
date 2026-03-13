from torch.utils.data import DataLoader
from retrieval_dataset import RetrievalDataset


def build_dataloader(
    train_path,
    val_path,
    user2idx,
    item2idx,
    root2idx,
    leaf2idx,
    batch_size=1024
):

    train_dataset = RetrievalDataset(
        train_path,
        user2idx,
        item2idx,
        root2idx,
        leaf2idx
    )

    val_dataset = RetrievalDataset(
        val_path,
        user2idx,
        item2idx,
        root2idx,
        leaf2idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader