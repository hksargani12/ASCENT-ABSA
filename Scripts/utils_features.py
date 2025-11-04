"""Utility functions for feature preparation and Integrated Gradients support."""

import torch
import numpy as np

def example2feature(example, max_seq_length, tokenizer):
    """
    Convert a single input example into tokenized BERT features.

    Args:
        example: tuple (text, gold_obj, label)
        max_seq_length: int, maximum sequence length
        tokenizer: BERT tokenizer

    Returns:
        features: dict with input_ids, input_mask, segment_ids, baseline_ids
        tokens_info: dict containing tokens, gold_obj, label, placeholders for predictions
    """
    ori_tokens = tokenizer.tokenize(example[0])
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    # Add [CLS] and [SEP] tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)

    # Convert to IDs and create attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)

    # Padding
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    # Sanity checks
    assert len(input_ids) == len(input_mask) == len(segment_ids) == len(baseline_ids) == max_seq_length

    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'baseline_ids': baseline_ids,
    }
    tokens_info = {
        "tokens": tokens,
        "label": example[2],
        "gold_obj": example[1],
        "pred_obj": None,
        "pred_label_prob": None
    }
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    """
    Generate scaled input points along the path between baseline and embedding for IG.

    Args:
        emb: Tensor of shape (1, ffn_size)
        batch_size: number of steps per batch
        num_batch: number of batches along the path

    Returns:
        res: Tensor of interpolated embeddings, shape (num_points, ffn_size)
        step: step vector between successive points
    """
    baseline = torch.zeros_like(emb)
    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points
    res = torch.cat([baseline + step * i for i in range(num_points)], dim=0)
    return res, step[0]


def convert_to_triplet_ig(ig_list, threshold=0.1):
    """
    Convert IG matrix into triplets of (layer, neuron, importance value),
    filtering by a relative threshold.

    Args:
        ig_list: list or np.ndarray, shape [num_layers, num_neurons]
        threshold: relative cutoff based on maximum IG value

    Returns:
        list of [layer_idx, neuron_idx, value]
    """
    ig = np.array(ig_list)
    max_ig = ig.max()
    ig_triplet = [
        [i, j, ig[i][j]]
        for i in range(ig.shape[0])
        for j in range(ig.shape[1])
        if ig[i][j] >= max_ig * threshold
    ]
    return ig_triplet
