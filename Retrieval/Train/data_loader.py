from torch.utils.data import DataLoader

from Retrieval.Train.retrieval_dataset import RetrievalDataset


def build_dataloader(
    train_path,
    val_path,
    user2idx,
    item2idx,
    root2idx,
    leaf2idx,
    batch_size=1024,
    num_workers=2,
    pin_memory=True,
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
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
