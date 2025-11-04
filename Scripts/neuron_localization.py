"""
BERT MLM runner
"""

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time

from torch.utils.data import DataLoader, Dataset
import pandas as pd

import transformers
from transformers import BertTokenizer, AutoConfig
from custom_bert import BertForMaskedLM
from utils_features import example2feature, scaled_input, convert_to_triplet_ig
import torch.nn.functional as F
import torch.nn as nn

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main(arg):
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_prefix",
                        default=None,
                        type=str,
                        required=True,
                        help="The output prefix to indentify each running of experiment. ")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--debug",
                        type=int,
                        default=-1,
                        help="How many examples to debug. -1 denotes no debugging")

    # parameters about integrated grad
    parser.add_argument("--get_pred",
                        action='store_true',
                        help="Whether to get prediction results.")
    parser.add_argument("--get_ig_pred",
                        action='store_true',
                        help="Whether to get integrated gradient at the predicted label.")
    parser.add_argument("--get_ig_gold",
                        action='store_true',
                        help="Whether to get integrated gradient at the gold label.")
    parser.add_argument("--get_base",
                        action='store_true',
                        help="Whether to get base values. ")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")

    # parse arguments
    args = parser.parse_args(arg)

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, args.output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Load the pre-trained BERT model
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    num_labels = 2  # 'positive' and 'negative'
    config = AutoConfig.from_pretrained(args.bert_model)
    model = BertForMaskedLM.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)

    # Data parallel if needed
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # set the model for the evaluation
    model.eval()
    print("Neuron localizations Started......................................................")

    # prepare eval set
    with open(args.data_path, 'r') as f:
        eval_bag_list_all = json.load(f)

    #print("Loaded data structure:", eval_bag_list_all[0])

    eval_bag_list_perrel = []
    for bag_idx, eval_bag in enumerate(eval_bag_list_all):
        eval_bag_list_perrel.append(eval_bag)

    # evaluate each sentiment text
    # record running time
    tic = time.perf_counter()
    print('start processing, while dataset is ',len(eval_bag_list_perrel))
    count = 0
    with jsonlines.open(os.path.join(args.output_dir, args.output_prefix + '.sentiment' + '.jsonl'), 'w') as fw:
        for bag_idx, sentiment_texts in enumerate(eval_bag_list_perrel):
            res_dict_bag = []

            #print("Priv_texts structure:", priv_texts)
            sum_list = [[0 for i in range(config.intermediate_size)] for j in range(config.num_hidden_layers)]
            sum_l = [[0 for i in range(config.intermediate_size)] for j in range(config.num_hidden_layers)]
            _, tokens_info = example2feature(sentiment_texts[0], args.max_seq_length, tokenizer)
            #print("Tokenized:", tokens_info['tokens'])

            for eval_example in sentiment_texts:
                #print(eval_example)
                eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)

                # Obtain the original masked word and label
                masked_word = eval_example[1]
                label = eval_example[2]

                #print("Tokenized:", tokens_info['tokens'])

                # convert features to long type tensors
                baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
                baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
                input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
                baseline_ids = baseline_ids.to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_tensor = torch.tensor(label, dtype=torch.long).to(device)

                # record real input length
                input_len = int(input_mask[0].sum())

                # record [MASK]'s position
                tgt_pos = tokens_info['tokens'].index('[MASK]')

                # record various results
                res_dict = {
                    'pred': [],
                    'ig_pred': [],
                    'ig_gold': [],
                    'base': []
                }

                # original pred prob
                # Prepare tokens_info['pred_obj'] and other data structures before loop
                tokens_info['pred_obj'] = {}

                # Loop over all layers to get ffn_weights, logits, and sentiment probabilities
                for tgt_layer in range(model.bert.config.num_hidden_layers):
                    # Run model to get the predictions
                    _, _, logits, sentiment_probs, ffn_weights = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)

                    # Obtain sentiment probabilities from the classifier on the final layer's hidden states
                    if tgt_layer == model.bert.config.num_hidden_layers - 1 and args.get_pred:
                        # print()
                        # print("Getting the sentiment probabilities.")
                        # print()
                        # Get sentiment probabilities from the final layer
                        sentiment_prob = sentiment_probs.squeeze()
                        tokens_info['pred_label_prob'] = sentiment_prob.tolist()

                    # Get the predicted token based on the logits
                    pred_label = int(torch.argmax(logits[0, :]))  # scalar
                    gold_label = tokenizer.convert_tokens_to_ids(tokens_info['gold_obj'])

                    # Convert top 10 predictions' indices to tokens
                    topk_values, topk_indices = torch.topk(logits[0, :], 10)  # shape of topk_indices and topk_values: (10,)
                    topk_tokens = tokenizer.convert_ids_to_tokens(topk_indices.tolist())

                    # Store the predicted tokens for the current layer
                    tokens_info['pred_obj'] = topk_tokens

                    # Perform Integrated Gradients (IG) for predicted label
                    if args.get_ig_pred:
                        ig_pred = None
                        scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)
                        scaled_weights.requires_grad_(True)

                        for batch_idx in range(args.num_batch):
                            batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                            _, _, _, _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=pred_label)
                            grad = grad.sum(dim=0)  # (ffn_size)
                            ig_pred = grad if ig_pred is None else torch.add(ig_pred, grad)  # (ffn_size)

                        ig_pred = ig_pred * weights_step  # (ffn_size)
                        res_dict['ig_pred'].append(ig_pred.tolist())

                    # Perform Integrated Gradients (IG) for gold label
                    if args.get_ig_gold:
                        ig_gold = None
                        for batch_idx in range(args.num_batch):
                            batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                            _, _, _, _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)
                            grad = grad.sum(dim=0)  # (ffn_size)
                            ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (ffn_size)

                        ig_gold = ig_gold * weights_step  # (ffn_size)
                        res_dict['ig_gold'].append(ig_gold.tolist())

                    # Store ffn_weights for each layer
                    if args.get_base:
                        res_dict['base'].append(ffn_weights.squeeze().tolist())

                # Sum Integrated Gradients for gold label and prediction across layers
                for i in range(len(res_dict['ig_gold'])):
                    for j in range(len(res_dict['ig_gold'][i])):
                        sum_list[i][j] += res_dict['ig_gold'][i][j]

                for i in range(len(res_dict['ig_pred'])):
                    for j in range(len(res_dict['ig_pred'][i])):
                        sum_l[i][j] += res_dict['ig_pred'][i][j]

                # Convert Integrated Gradient results to triplets if necessary
                if args.get_ig_pred:
                    res_dict['ig_pred'] = convert_to_triplet_ig(res_dict['ig_pred'])
                if args.get_ig_gold:
                    res_dict['ig_gold'] = convert_to_triplet_ig(res_dict['ig_gold'])
                if args.get_base:
                    res_dict['base'] = convert_to_triplet_ig(res_dict['base'])

                # Final result
                res_dict_bag.append([tokens_info, res_dict])
                fw.write(res_dict_bag)

    # record running time
    toc = time.perf_counter()
    print(f"***** Sentiment: evaluated. Costing time: {toc - tic:0.4f} seconds *****")
def run():
    # Example arguments
    arg = [
        '--data_path', 'masked_laptop_train.json',
        '--bert_model', 'bert-base-uncased',
        '--output_dir', 'outputs_neurons',
        '--output_prefix', 'neurons_laptop',
        '--do_lower_case',
        '--gpus', '0',
        '--max_seq_length', '128',
        '--get_pred',
        '--get_ig_pred',
        '--get_ig_gold',
        '--batch_size', '20',
        '--num_batch', '1',
        '--debug', '100000'
    ]
    main(arg)

if __name__ == "__main__":
    run()