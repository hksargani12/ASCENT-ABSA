import torch
import pandas as pd
from torch.utils.data import Dataset

def preprocess_data_from_csv(df, tokenizer, max_seq_length):
    input_ids, attention_masks, segment_ids, baseline_ids, labels, tgt_positions, gold_objs = [], [], [], [], [], [], []

    for _, row in df.iterrows():
        masked_sentence = row['Masked Sentence']  # The sentence with [MASK] token
        masked_word = row['Masked Word']  # The actual word that was masked (gold_obj)
        sentiment = row['Sentiment']  # Sentiment label

        # Tokenize the masked sentence
        tokens = tokenizer.tokenize(masked_sentence)

        # Truncate to max sequence length (keeping space for [CLS] and [SEP])
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]

        # Add [CLS] and [SEP]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids_row = [0] * len(tokens)

        # Replace tokens with token ids
        input_ids_row = tokenizer.convert_tokens_to_ids(tokens)
        baseline_ids_row = [tokenizer.convert_tokens_to_ids(["[UNK]"])[0]] * len(tokens)
        attention_mask_row = [1] * len(input_ids_row)

        # Pad sequences to max_seq_length
        padding_length = max_seq_length - len(input_ids_row)
        input_ids_row += [0] * padding_length
        baseline_ids_row += [0] * padding_length
        segment_ids_row += [0] * padding_length
        attention_mask_row += [0] * padding_length

        # Find the position of the [MASK] token (tgt_pos)
        try:
            tgt_pos = tokens.index('[MASK]')
        except ValueError:
            tgt_pos = -1  # Mark as -1 if no [MASK] token is found

        # Append the processed data
        input_ids.append(torch.tensor(input_ids_row, dtype=torch.long))
        baseline_ids.append(torch.tensor(baseline_ids_row, dtype=torch.long))
        attention_masks.append(torch.tensor(attention_mask_row, dtype=torch.long))
        segment_ids.append(torch.tensor(segment_ids_row, dtype=torch.long))
        labels.append(torch.tensor(sentiment, dtype=torch.long))  # Sentiment label
        tgt_positions.append(tgt_pos)  # Position of [MASK] token
        gold_objs.append(masked_word)  # The actual masked word

    return (
        torch.stack(input_ids), torch.stack(baseline_ids), torch.stack(attention_masks),
        torch.stack(segment_ids), torch.stack(labels), torch.tensor(tgt_positions), gold_objs
    )

class CustomDataset(Dataset):
    def __init__(self, input_ids, baseline_ids, attention_masks, segment_ids, labels, tgt_positions, gold_objs):
        self.input_ids = input_ids
        self.baseline_ids = baseline_ids
        self.attention_masks = attention_masks
        self.segment_ids = segment_ids
        self.labels = labels
        self.tgt_positions = tgt_positions
        self.gold_objs = gold_objs

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'baseline_ids': self.baseline_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'segment_ids': self.segment_ids[idx],
            'label': self.labels[idx],
            'tgt_pos': self.tgt_positions[idx],
            'gold_obj': self.gold_objs[idx]
        }