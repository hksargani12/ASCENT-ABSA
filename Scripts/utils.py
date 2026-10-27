"""Utility functions and auxiliary loss modules for ASCENT framework."""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss implementation.
    Reference:
        Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
        https://arxiv.org/pdf/2004.11362.pdf
    Supports both supervised and unsupervised (SimCLR) modes.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: Tensor of shape [batch_size, n_views, feature_dim]
            labels: Tensor of shape [batch_size], optional
            mask: Binary mask [batch_size, batch_size], optional
        Returns:
            Scalar loss value.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("Expected [batch_size, n_views, feature_dim] input.")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`.")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Label count does not match batch size.")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode}")

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask setup
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask *= logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-30)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class LabelSmoothLoss(nn.Module):
    """Cross-entropy loss with label smoothing."""
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


_FFN_IN = re.compile(r"encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)$")
_FFN_OUT = re.compile(r"encoder\.layer\.(\d+)\.output\.dense\.weight$")


def build_neuron_masks(model, selected_neurons, train_prefixes=()):
    """Freeze everything except the selected FFN units and return {param: 0/1 mask}."""
    masks = {}
    for name, p in model.named_parameters():
        m_in, m_out = _FFN_IN.search(name), _FFN_OUT.search(name)
        m = m_in or m_out
        idx = selected_neurons.get(m.group(1)) if m else None

        if not idx:
            # train_prefixes keeps the task head trainable when it is not part of the FFN
            p.requires_grad = any(name.startswith(k) for k in train_prefixes)
            continue

        idx = torch.as_tensor(idx, dtype=torch.long, device=p.device)
        dim = p.shape[1] if m_out else p.shape[0]
        assert int(idx.max()) < dim, f"{name}: neuron index {int(idx.max())} out of range ({dim})"

        mask = torch.zeros_like(p)
        if m_out:
            mask[:, idx] = 1.0  # outgoing weights of unit u (columns of W2)
        else:
            mask[idx] = 1.0     # incoming weights and bias of unit u (rows of W1)
        p.requires_grad = True
        masks[p] = mask
    return masks


def apply_masks(masks):
    """Zero the gradient of every non-selected entry before the optimizer step."""
    for p, mask in masks.items():
        if p.grad is not None:
            p.grad.mul_(mask)
