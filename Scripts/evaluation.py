"""Evaluation utilities for ASCENT framework."""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def evaluate_model(model, test_loader, device, vocab_size, num_labels, tokenizer):
    """
    Evaluate the ASCENT model on the test set.

    Computes metrics for:
        1. Sentiment classification
        2. Masked Language Modeling (MLM) – accuracy and perplexity

    Args:
        model: Trained model instance.
        test_loader: DataLoader for the evaluation set.
        device: torch.device('cpu' or 'cuda').
        vocab_size: Vocabulary size for MLM.
        num_labels: Number of sentiment labels.
        tokenizer: Tokenizer used for data preprocessing.

    Returns:
        dict: Aggregated evaluation metrics.
    """
    model.eval()
    all_sentiment_preds, all_sentiment_labels = [], []
    all_mlm_preds, all_mlm_labels = []
    total_neg_log_likelihood, total_mlm_tokens = 0.0, 0

    with torch.no_grad():
        for batch in test_loader:
            # Load batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            labels = batch['label'].to(device)
            tgt_pos = batch['tgt_pos'].to(device)
            gold_obj_words = batch['gold_obj']

            # Convert gold target words to token IDs (ensure one token per word)
            gold_obj_ids = []
            for word in gold_obj_words:
                token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                if len(token_ids) == 1:
                    gold_obj_ids.append(token_ids[0])
                else:
                    gold_obj_ids.append(tokenizer.convert_tokens_to_ids([word])[0])
            gold_obj = torch.tensor(gold_obj_ids, dtype=torch.long).to(device)

            # Forward pass
            last_hidden, _, tgt_logits, sentiment_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=segment_ids,
                tgt_pos=tgt_pos,
                tgt_layer=0
            )

            # Sentiment prediction
            _, sentiment_preds = torch.max(sentiment_logits, dim=1)
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_sentiment_labels.extend(labels.cpu().numpy())

            # MLM predictions and perplexity
            gold_obj = gold_obj.unsqueeze(1).expand(-1, tgt_logits.size(1)).contiguous()
            tgt_logits = tgt_logits.view(-1, tgt_logits.size(-1))
            gold_obj = gold_obj.view(-1)

            vocab_size = tgt_logits.size(-1)
            gold_obj = torch.clamp(gold_obj, min=0, max=vocab_size - 1)

            _, mlm_preds = torch.max(tgt_logits, dim=1)
            all_mlm_preds.extend(mlm_preds.cpu().numpy())
            all_mlm_labels.extend(gold_obj.cpu().numpy())

            log_probs = F.log_softmax(tgt_logits, dim=-1)
            gold_log_probs = log_probs[torch.arange(gold_obj.size(0)), gold_obj]
            total_neg_log_likelihood += -gold_log_probs.sum().item()
            total_mlm_tokens += gold_obj.size(0)

    # Sentiment metrics
    sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_sentiment_labels, all_sentiment_preds, average='macro'
    )

    # MLM metrics
    mlm_correct = (np.array(all_mlm_preds) == np.array(all_mlm_labels)).sum()
    mlm_total = len(all_mlm_labels)
    mlm_accuracy = mlm_correct / mlm_total
    mlm_perplexity = np.exp(total_neg_log_likelihood / total_mlm_tokens)

    metrics = {
        "sentiment_accuracy": sentiment_accuracy,
        "sentiment_precision_macro": precision,
        "sentiment_recall_macro": recall,
        "sentiment_f1_macro": f1,
        "mlm_accuracy": mlm_accuracy,
        "mlm_perplexity": mlm_perplexity
    }

    return metrics
