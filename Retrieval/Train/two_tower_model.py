import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):

    def __init__(
        self,
        num_users,
        num_items,
        num_roots,
        num_leafs,
        embed_dim=64,
        num_user_numeric=6,
        num_item_numeric=5
    ):
        super().__init__()

        # User embedding
        self.user_embedding = nn.Embedding(num_users + 1, embed_dim)

        # Shared item embedding (dùng cho cả history và target item)
        self.item_embedding = nn.Embedding(
            num_items + 1,
            embed_dim,
            padding_idx=0
        )

        # Category embeddings
        self.root_embedding = nn.Embedding(num_roots + 1, 16)
        self.leaf_embedding = nn.Embedding(num_leafs + 1, 16)

        # USER TOWER
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + num_user_numeric, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

        # ITEM TOWER
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim + 16 + 16 + num_item_numeric, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, batch):

        user_id = batch["user_id"]
        item_id = batch["item_id"]
        recent_items = batch["recent_items"]

        user_numeric = batch["user_numeric"]
        item_numeric = batch["item_numeric"]

        root = batch["root"]
        leaf = batch["leaf"]

        # ===== USER TOWER =====

        user_emb = self.user_embedding(user_id)

        # dùng chung item_embedding cho history
        recent_emb = self.item_embedding(recent_items)

        # mask để bỏ padding
        mask = (recent_items != 0).float()

        recent_emb = recent_emb * mask.unsqueeze(-1)

        recent_emb = recent_emb.sum(dim=1) / (
            mask.sum(dim=1, keepdim=True) + 1e-8
        )

        user_features = torch.cat(
            [user_emb, recent_emb, user_numeric],
            dim=1
        )

        user_vec = self.user_mlp(user_features)

        # ===== ITEM TOWER =====

        item_emb = self.item_embedding(item_id)

        root_emb = self.root_embedding(root)
        leaf_emb = self.leaf_embedding(leaf)

        item_features = torch.cat(
            [item_emb, root_emb, leaf_emb, item_numeric],
            dim=1
        )

        item_vec = self.item_mlp(item_features)

        return user_vec, item_vec