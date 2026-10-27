#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASCENT fine-tuning (full or neuron-masked)
Objective: sentiment polarity cross-entropy only (L_SP).
Modes:
  - full: standard fine-tuning
  - dormant: train only the dormant FFN neurons
  - salient: train only the salient FFN neurons
  - random: train only a random FFN neuron subset (precomputed JSON)

In masked modes a neuron owns its incoming row of intermediate.dense (weight and bias)
and its outgoing column of output.dense. All other parameters stay frozen.

Datasets: lap14, res14, mams
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoConfig

from preprocess import preprocess_data_from_csv, CustomDataset
from custom_bert import BertForMaskedLM
from utils import build_neuron_masks, apply_masks
from evaluation import evaluate_model


def setup_logger(log_dir: str, filename: str = "finetune.log", level: str = "INFO", to_console: bool = False):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("ASCENT_FT")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, filename))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)
    return logger


def set_seed(seed: int, n_gpu: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def resolve_neuron_file(mode: str, dataset: str, neurons_dir: str | None, neurons_file: str | None) -> str | None:
    if mode == "full":
        return None
    if neurons_file:
        return neurons_file
    base = neurons_dir if neurons_dir else "./neurons"
    return os.path.join(base, f"neurons_{dataset}_{mode}.json")


def train_one_epoch(model, optimizer, train_loader, device, args, masks):
    model.train()

    total_loss = 0.0
    cls_correct = 0
    cls_total = 0
    start = time.time()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["segment_ids"].to(device)
        tgt_pos = batch["tgt_pos"].to(device)
        labels = batch["label"].to(device)

        # The model forward still expects tgt_pos; only the sentiment logits enter the loss
        _, _, _, sentiment_logits, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            tgt_pos=tgt_pos,
            tgt_layer=0
        )

        # L_SP: sentiment polarity cross-entropy (plain unless --label_smoothing > 0)
        loss = F.cross_entropy(
            sentiment_logits.view(-1, args.num_labels),
            labels.view(-1),
            label_smoothing=args.label_smoothing
        )

        optimizer.zero_grad()
        loss.backward()
        if masks:
            apply_masks(masks)
        optimizer.step()

        total_loss += float(loss.item())
        pred_labels = sentiment_logits.argmax(dim=1)
        cls_correct += (pred_labels == labels).sum().item()
        cls_total += labels.size(0)

    return {
        "train_sp_loss": total_loss / max(1, len(train_loader)),
        "train_cls_acc": cls_correct / max(1, cls_total),
        "epoch_time_min": (time.time() - start) / 60.0,
    }


def main():
    p = argparse.ArgumentParser(description="ASCENT fine-tuning (full or neuron-masked)")
    p.add_argument("--dataset_name", type=str, required=True, choices=["lap14", "res14", "mams"])
    p.add_argument("--train_data_path", type=str, required=True)
    p.add_argument("--test_data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--output_prefix", type=str, default="ascent_ft")
    p.add_argument("--overwrite_output_dir", action="store_true")
    p.add_argument("--bert_model", type=str, required=True)
    p.add_argument("--pretrained_checkpoint", type=str, default=None, help="Path to .pt state_dict to load before FT")
    p.add_argument("--do_lower_case", action="store_true")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--num_labels", type=int, default=3)
    p.add_argument("--train_epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--save_each_epoch", action="store_true")
    p.add_argument("--finetune_mode", type=str, required=True, choices=["full", "dormant", "salient", "random"])
    p.add_argument("--neurons_dir", type=str, default=None, help="Base dir for neuron JSONs")
    p.add_argument("--neurons_file", type=str, default=None, help="Explicit neuron file path; overrides neurons_dir")
    p.add_argument("--train_prefixes", type=str, default="",
                   help="Comma-separated parameter-name prefixes kept trainable outside the FFN (e.g. the task head)")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log_to_console", action="store_true")

    args = p.parse_args()

    out_dir = os.path.join(args.output_dir, args.dataset_name, args.finetune_mode)
    if os.path.exists(out_dir) and os.listdir(out_dir) and not args.overwrite_output_dir:
        raise FileExistsError(f"Output directory not empty: {out_dir}. Use --overwrite_output_dir to proceed.")
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logger(out_dir, "finetune.log", args.log_level, args.log_to_console)
    with open(os.path.join(out_dir, f"{args.output_prefix}.args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    else:
        device = torch.device(f"cuda:{args.gpus}")
        n_gpu = 1
    set_seed(args.seed, n_gpu)
    logger.info(f"Dataset={args.dataset_name} | Mode={args.finetune_mode} | Device={device}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_df = pd.read_csv(args.train_data_path)
    test_df = pd.read_csv(args.test_data_path)

    tr = preprocess_data_from_csv(train_df, tokenizer, args.max_seq_length)
    te = preprocess_data_from_csv(test_df, tokenizer, args.max_seq_length)

    train_dataset = CustomDataset(*tr)
    test_dataset = CustomDataset(*te)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    logger.info(f"Train samples={len(train_dataset)} | Test samples={len(test_dataset)}")

    torch.cuda.empty_cache()
    _ = AutoConfig.from_pretrained(args.bert_model)
    model = BertForMaskedLM.from_pretrained(args.bert_model, num_labels=args.num_labels).to(device)

    if args.pretrained_checkpoint and os.path.isfile(args.pretrained_checkpoint):
        state = torch.load(args.pretrained_checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained checkpoint: {args.pretrained_checkpoint}")

    # Masks must exist before the optimizer so that frozen tensors never enter it
    masks = None
    if args.finetune_mode != "full":
        nf = resolve_neuron_file(args.finetune_mode, args.dataset_name, args.neurons_dir, args.neurons_file)
        if not nf or not os.path.isfile(nf):
            raise FileNotFoundError(f"Neuron file not found for mode={args.finetune_mode}: {nf}")
        with open(nf, "r") as f:
            selected_neurons = json.load(f)
        logger.info(f"Using neuron file: {nf}")

        train_prefixes = [s for s in args.train_prefixes.split(",") if s]
        masks = build_neuron_masks(model, selected_neurons, train_prefixes)
        n_active = int(sum(m.sum().item() for m in masks.values()))
        n_total = sum(p_.numel() for p_ in model.parameters())
        logger.info(f"Trainable FFN entries={n_active} of {n_total} ({100.0 * n_active / n_total:.3f}%)")

    # L2 decay adds a gradient to masked entries, so it is off in masked modes
    wd = args.weight_decay if masks is None else 0.0
    optimizer = Adam([p_ for p_ in model.parameters() if p_.requires_grad],
                     lr=args.learning_rate, weight_decay=wd)

    metrics_path = os.path.join(out_dir, "metrics.json")
    all_metrics = []

    for epoch in range(1, args.train_epochs + 1):
        tr_metrics = train_one_epoch(model, optimizer, train_loader, device, args, masks)

        eval_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            vocab_size=model.config.vocab_size,
            num_labels=args.num_labels,
            tokenizer=tokenizer
        )

        row = {
            "epoch": epoch,
            **tr_metrics,
            **{f"eval_{k}": v for k, v in eval_metrics.items()}
        }
        all_metrics.append(row)
        logger.info(
            f"Epoch {epoch}/{args.train_epochs} | "
            f"train_sp={tr_metrics['train_sp_loss']:.4f} "
            f"train_acc={tr_metrics['train_cls_acc']:.4f} "
            f"| eval_sent_acc={eval_metrics.get('sentiment_accuracy', 0):.4f} "
            f"eval_f1_macro={eval_metrics.get('sentiment_f1_macro', 0):.4f} "
            f"| time={tr_metrics['epoch_time_min']:.2f}m"
        )
        if args.save_each_epoch:
            ckpt = os.path.join(out_dir, f"{args.output_prefix}.epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")

        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

    final_ckpt = os.path.join(out_dir, f"{args.output_prefix}.final.pt")
    torch.save(model.state_dict(), final_ckpt)

    summary = {
        "run_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset_name,
        "mode": args.finetune_mode,
        "epochs": args.train_epochs,
        "final_checkpoint": final_ckpt,
        "metrics_json": metrics_path,
        "args_json": os.path.join(out_dir, f"{args.output_prefix}.args.json"),
    }
    with open(os.path.join(out_dir, f"{args.output_prefix}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Finished. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()