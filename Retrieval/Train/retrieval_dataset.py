import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class RetrievalDataset(Dataset):

    def __init__(
        self,
        data_path,
        user2idx,
        item2idx,
        root2idx,
        leaf2idx,
        max_seq_len=5
    ):

        self.df = pd.read_parquet(data_path)

        self.user2idx = user2idx
        self.item2idx = item2idx
        self.root2idx = root2idx
        self.leaf2idx = leaf2idx

        self.max_seq_len = max_seq_len

        # JSON object keys are always strings after loading.
        # Convert all ID maps to string-keyed dicts so mapping works
        # whether source columns are int or str.
        self.user2idx = {str(k): int(v) for k, v in user2idx.items()}
        self.item2idx = {str(k): int(v) for k, v in item2idx.items()}
        self.root2idx = {str(k): int(v) for k, v in root2idx.items()}
        self.leaf2idx = {str(k): int(v) for k, v in leaf2idx.items()}

        self._prepare()

    def _pad_seq(self, seq):

        seq = seq[:self.max_seq_len]

        if len(seq) < self.max_seq_len:
            seq = seq + [0] * (self.max_seq_len - len(seq))

        return seq

    def _prepare(self):

        self.user_raw_ids = self.df["user_id"].values
        self.item_raw_ids = self.df["item_id"].values

        # map user_id -> index
        self.df["user_idx"] = (
            self.df["user_id"].astype(str)
            .map(self.user2idx)
            .fillna(0)
            .astype(int)
        )

        # map item_id -> index
        self.df["item_idx"] = (
            self.df["item_id"].astype(str)
            .map(self.item2idx)
            .fillna(0)
            .astype(int)
        )

        # map category root / leaf
        self.df["root_idx"] = (
            self.df["root"].astype(str)
            .map(self.root2idx)
            .fillna(0)
            .astype(int)
        )

        self.df["leaf_idx"] = (
            self.df["leaf"].astype(str)
            .map(self.leaf2idx)
            .fillna(0)
            .astype(int)
        )

        # encode recent_items + padding
        self.df["recent_items"] = self.df["recent_items"].apply(
            lambda seq: self._pad_seq(
                [
                    self.item2idx.get(str(i), 0)
                    for i in (seq if isinstance(seq, list) else [])
                ]
            )
        )

        # convert id columns
        self.users = self.df["user_idx"].values.astype(np.int64)
        self.items = self.df["item_idx"].values.astype(np.int64)

        self.roots = self.df["root_idx"].values.astype(np.int64)
        self.leafs = self.df["leaf_idx"].values.astype(np.int64)

        # convert sequence
        self.recent_items = np.array(
            self.df["recent_items"].tolist(),
            dtype=np.int64
        )

        # user numeric features
        self.user_numeric = self.df[
            [
                "total_views",
                "total_addtocart",
                "total_transactions",
                "unique_items",
                "addtocart_rate",
                "purchase_rate",
            ]
        ].fillna(0).values.astype(np.float32)

        # item numeric features
        self.item_numeric = self.df[
            [
                "total_views_item",
                "total_addtocart_item",
                "total_transactions_item",
                "cart_rate",
                "purchase_rate_item",
            ]
        ].fillna(0).values.astype(np.float32)

        # event weight
        if "weight" in self.df.columns:
            self.weights = self.df["weight"].values.astype(np.float32)
        else:
            self.weights = np.ones(len(self.df), dtype=np.float32)

        if "event" in self.df.columns:
            self.events = self.df["event"].astype(str).values
            self.relevance = np.isin(
                self.events,
                ["addtocart", "transaction"]
            ).astype(np.float32)
        else:
            self.events = np.array(["unknown"] * len(self.df), dtype=object)
            self.relevance = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        return {
            "user_id": torch.tensor(self.users[idx], dtype=torch.long),
            "item_id": torch.tensor(self.items[idx], dtype=torch.long),

            "recent_items": torch.tensor(
                self.recent_items[idx],
                dtype=torch.long
            ),

            "user_numeric": torch.tensor(
                self.user_numeric[idx],
                dtype=torch.float32
            ),

            "item_numeric": torch.tensor(
                self.item_numeric[idx],
                dtype=torch.float32
            ),

            "root": torch.tensor(self.roots[idx], dtype=torch.long),
            "leaf": torch.tensor(self.leafs[idx], dtype=torch.long),

            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
            "relevance": torch.tensor(self.relevance[idx], dtype=torch.float32),
            "user_raw_id": str(self.user_raw_ids[idx]),
            "item_raw_id": str(self.item_raw_ids[idx]),
            "event": self.events[idx],
        }
