from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class FisherSnapshot:
    fisher: dict[str, torch.Tensor]
    params: dict[str, torch.Tensor]


def compute_fisher_information(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
    max_batches: int = 200,
) -> FisherSnapshot:
    model.eval()
    fisher: dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters() if p.requires_grad
    }
    n_seen = 0
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0].to(device), batch[1].to(device)
        else:
            inputs = batch.to(device)
            targets = None
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        if targets is None:
            log_probs = F.log_softmax(outputs, dim=-1)
            sampled = log_probs.exp().multinomial(1).squeeze(-1)
            loss = F.nll_loss(log_probs, sampled)
        else:
            loss = F.cross_entropy(outputs, targets.long())
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            fisher[n] += p.grad.detach() ** 2
        n_seen += 1
    if n_seen > 0:
        for n in fisher:
            fisher[n] /= float(n_seen)
    params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    return FisherSnapshot(fisher=fisher, params=params)


def ewc_penalty(
    model: nn.Module,
    snapshot: FisherSnapshot,
    *,
    lambda_ewc: float = 5000.0,
) -> torch.Tensor:
    if not snapshot.fisher:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n not in snapshot.fisher:
            continue
        fisher = snapshot.fisher[n]
        prior = snapshot.params[n]
        loss = loss + (fisher * (p - prior) ** 2).sum()
    return 0.5 * lambda_ewc * loss


def total_loss_with_ewc(
    task_loss: torch.Tensor,
    model: nn.Module,
    snapshot: FisherSnapshot | None,
    *,
    lambda_ewc: float = 5000.0,
) -> torch.Tensor:
    if snapshot is None:
        return task_loss
    return task_loss + ewc_penalty(model, snapshot, lambda_ewc=lambda_ewc)
