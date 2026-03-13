import torch
import torch.nn.functional as F


def retrieval_loss(user_vec, item_vec, weight, temperature=0.05):

    # normalize embeddings
    user_vec = F.normalize(user_vec, dim=1)
    item_vec = F.normalize(item_vec, dim=1)

    logits = torch.matmul(user_vec, item_vec.t()) / temperature

    labels = torch.arange(len(user_vec)).to(user_vec.device)

    loss = F.cross_entropy(logits, labels, reduction="none")

    loss = loss * weight

    return loss.mean()