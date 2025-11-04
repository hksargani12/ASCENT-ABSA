#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASCENT Pretraining Runner
- Performs pretraining with supervised contrastive and classification objectives.
- Supports multiple datasets (e.g., Amazon, Yelp) and saves outputs accordingly.
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

from custom_bert import BertForMaskedLM
from utils import SupConLoss
from preprocess import preprocess_data_from_csv, CustomDataset


# ---------- LOGGING ----------
def setup_logging(output_dir, log_filename="train.log", log_level="INFO", log_to_console=False):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("bert_mlm_pretrain")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(output_dir, log_filename))
    fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)
    return logger


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


# ---------- ARGUMENTS ----------
def build_arg_parser():
    parser = argparse.ArgumentParser(description="ASCENT Pretraining Runner (Amazon/Yelp)")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training CSV file.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["amazon", "yelp"],
                        help="Dataset name (used for saving weights and logs).")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory.")
    parser.add_argument("--output_prefix", type=str, default="pretrain_run", help="Prefix for output files.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Allow overwriting non-empty directory.")
    parser.add_argument("--save_each_epoch", action="store_true", help="Save model after every epoch.")

    parser.add_argument("--bert_model", type=str, required=True, help="Model name or path (e.g., bert-base-uncased).")
    parser.add_argument("--do_lower_case", action="store_true", help="Use uncased tokenizer.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length for tokenization.")
    parser.add_argument("--train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels.")
    parser.add_argument("--lambda_scl", type=float, default=2.0, help="Weight for SupCon loss.")
    parser.add_argument("--lambda_mlm", type=float, default=1.0, help="Weight for MLM loss.")
    parser.add_argument("--lambda_cls", type=float, default=1.0, help="Weight for classification loss.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--gpus", type=str, default="0", help="GPU ID (single GPU).")
    parser.add_argument("--no_cuda", action="store_true", help="Use CPU only.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_to_console", action="store_true", help="Also log to console.")
    parser.add_argument("--progress_splits", type=int, default=10, help="Log progress n times per epoch.")
    return parser


# ---------- TRAINING ----------
def train(args, logger):
    # Prepare dataset-specific output folder
    dataset_out_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    args_path = os.path.join(dataset_out_dir, f"{args.output_prefix}_{args.dataset_name}.args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    logger.info(f"Saved args: {args_path}")

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    else:
        device = torch.device(f"cuda:{args.gpus}")
        n_gpu = 1
    logger.info(f"Using device: {device}")

    set_seed(args.seed, n_gpu)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_df = pd.read_csv(args.train_data_path)

    input_ids, baseline_ids, attention_masks, segment_ids, labels, tgt_positions, gold_objs = \
        preprocess_data_from_csv(train_df, tokenizer, args.max_seq_length)

    dataset = CustomDataset(input_ids, baseline_ids, attention_masks, segment_ids, labels, tgt_positions, gold_objs)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

    config = AutoConfig.from_pretrained(args.bert_model)
    model = BertForMaskedLM.from_pretrained(args.bert_model, num_labels=args.num_labels).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastive_loss = SupConLoss()

    total_samples = len(loader.dataset)
    step_size = max(1, total_samples // args.progress_splits)
    total_training_time = 0.0
    metrics = []

    logger.info(f"Starting pretraining on {args.dataset_name} ({len(loader)} batches per epoch)")

    for epoch in range(args.train_epochs):
        model.train()
        start = time.time()
        scl_sum, mlm_sum, cls_sum, total_sum = 0, 0, 0, 0
        cls_correct, cls_total = 0, 0
        mlm_correct, mlm_total = 0, 0
        samples_processed = 0

        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            tgt_pos = batch['tgt_pos'].to(device)
            labels = batch['label'].to(device)

            gold_obj_words = batch['gold_obj']
            gold_obj_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w))[0] for w in gold_obj_words]
            gold_obj = torch.tensor(gold_obj_ids, dtype=torch.long).to(device)

            last_hidden, _, tgt_logits, sentiment_prob, _ = model(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0
            )

            normed = F.normalize(last_hidden, dim=-1)
            scl_loss = contrastive_loss(normed.unsqueeze(1), labels=labels)
            mlm_loss = F.cross_entropy(tgt_logits.view(-1, tgt_logits.size(-1)),
                                       gold_obj.unsqueeze(1).expand(-1, tgt_logits.size(1)).reshape(-1))
            cls_loss = F.cross_entropy(sentiment_prob.view(-1, args.num_labels), labels.view(-1))
            loss = args.lambda_scl * scl_loss + args.lambda_mlm * mlm_loss + args.lambda_cls * cls_loss
            loss.backward()
            optimizer.step()

            scl_sum += scl_loss.item()
            mlm_sum += mlm_loss.item()
            cls_sum += cls_loss.item()
            total_sum += loss.item()

            _, pred_cls = torch.max(sentiment_prob, dim=1)
            cls_correct += (pred_cls == labels).sum().item()
            cls_total += labels.size(0)
            _, pred_mlm = torch.max(tgt_logits.view(-1, tgt_logits.size(-1)), dim=1)
            mlm_correct += (pred_mlm == gold_obj.unsqueeze(1).expand(-1, tgt_logits.size(1)).reshape(-1)).sum().item()
            mlm_total += gold_obj.size(0) * tgt_logits.size(1)

            samples_processed += input_ids.size(0)
            if samples_processed >= step_size:
                pct = (samples_processed / total_samples) * 100
                elapsed = (time.time() - start) / 60
                logger.info(f"[{args.dataset_name}] Epoch {epoch+1}/{args.train_epochs} | {pct:.1f}% done | {elapsed:.2f} min")
                samples_processed = 0

        epoch_time = (time.time() - start) / 60
        total_training_time += epoch_time
        cls_acc = cls_correct / max(1, cls_total)
        mlm_acc = mlm_correct / max(1, mlm_total)
        logger.info(f"[{args.dataset_name}] Epoch {epoch+1}: "
                    f"SCL {scl_sum/len(loader):.4f} | MLM {mlm_sum/len(loader):.4f} | CLS {cls_sum/len(loader):.4f} "
                    f"| Acc CLS {cls_acc:.4f} | Acc MLM {mlm_acc:.4f} | Time {epoch_time:.2f} min")

        if args.save_each_epoch:
            ckpt = os.path.join(dataset_out_dir, f"{args.output_prefix}_{args.dataset_name}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)
            logger.info(f"Saved checkpoint: {ckpt}")

        metrics.append({
            "epoch": epoch + 1,
            "scl_loss": scl_sum / len(loader),
            "mlm_loss": mlm_sum / len(loader),
            "cls_loss": cls_sum / len(loader),
            "cls_acc": cls_acc,
            "mlm_acc": mlm_acc,
            "epoch_time_min": epoch_time
        })

    df = pd.DataFrame(metrics)
    metrics_file = os.path.join(dataset_out_dir, f"{args.output_prefix}_{args.dataset_name}.metrics.csv")
    df.to_csv(metrics_file, index=False)
    final_ckpt = os.path.join(dataset_out_dir, f"{args.output_prefix}_{args.dataset_name}_final.pt")
    torch.save(model.state_dict(), final_ckpt)

    summary = {
        "dataset": args.dataset_name,
        "epochs": args.train_epochs,
        "total_training_time_min": total_training_time,
        "metrics_file": metrics_file,
        "final_checkpoint": final_ckpt
    }
    with open(os.path.join(dataset_out_dir, f"{args.output_prefix}_{args.dataset_name}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Finished pretraining on {args.dataset_name}. Total time: {total_training_time:.2f} min")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    logger = setup_logging(args.output_dir,
                           log_filename=f"{args.output_prefix}_{args.dataset_name}.log",
                           log_level=args.log_level,
                           log_to_console=args.log_to_console)
    train(args, logger)


if __name__ == "__main__":
    main()
