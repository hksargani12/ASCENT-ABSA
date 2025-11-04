"""Custom PyTorch implementation of BERT for ASCENT framework.
Includes:
  - Core BERT encoder and embeddings
  - Masked Language Modeling (MLM) head
  - Sentiment classification head
  - Layer-wise FFN access for interpretability and neuron analysis
"""

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import torch
from torch import nn
import torch.nn.functional as F
from transformers.file_utils import cached_path

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Pretrained model URLs and config names
# -------------------------------------------------------------------------
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'  # fine-tuned weights differ per dataset

# -------------------------------------------------------------------------
# Activation functions
# -------------------------------------------------------------------------
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
class BertConfig(object):
    """BERT model configuration container."""
    def __init__(self, vocab_size_or_config_json_file, hidden_size=768,
                 num_hidden_layers=12, num_attention_heads=12,
                 intermediate_size=3072, hidden_act="swish",
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512, type_vocab_size=2,
                 initializer_range=0.02):
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("Invalid argument type for config: must be int or str")

    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            return cls.from_dict(json.loads(reader.read()))

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# -------------------------------------------------------------------------
# LayerNorm, Embeddings, and Core Attention Modules
# -------------------------------------------------------------------------
class BertLayerNorm(nn.Module):
    """Implements LayerNorm in TensorFlow style (epsilon inside sqrt)."""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Constructs input embeddings from tokens, positions, and segments."""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeddings = (self.word_embeddings(input_ids)
                      + self.position_embeddings(position_ids)
                      + self.token_type_embeddings(token_type_ids))
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)

# -------------------------------------------------------------------------
# Transformer Encoder
# -------------------------------------------------------------------------
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("Hidden size must be multiple of attention heads.")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + attention_mask
        probs = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.all_head_size)
        return context, probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        att_output, att_score = self.self(input_tensor, attention_mask)
        return self.output(att_output, input_tensor), att_score


class BertIntermediate(nn.Module):
    """Feed-forward sublayer; includes interpretability operations."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.transform_tmp_score = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        hidden_states = self.act_fn(self.dense(hidden_states))
        if tmp_score is not None:
            hidden_states[:, tgt_pos, :] = self.transform_tmp_score(tmp_score)
        if imp_op == 'return':
            imp_weights = []
        if imp_pos is not None:
            for layer, pos in imp_pos:
                if imp_op == 'remove':
                    hidden_states[:, tgt_pos, pos] = 0.0
                elif imp_op == 'enhance':
                    hidden_states[:, tgt_pos, pos] *= 2.0
                elif imp_op == 'return':
                    imp_weights.append(hidden_states[0, tgt_pos, pos].item())
        return (hidden_states, imp_weights) if imp_op == 'return' else hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    """Full Transformer layer."""
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        att_out, att_score = self.attention(hidden_states, attention_mask)
        if imp_op == 'return':
            inter_out, imp_weights = self.intermediate(att_out, tgt_pos, tmp_score, imp_pos, imp_op)
        else:
            inter_out = self.intermediate(att_out, tgt_pos, tmp_score, imp_pos, imp_op)
        out = self.output(inter_out, att_out)
        return (out, inter_out, imp_weights) if imp_op == 'return' else (out, inter_out)


class BertEncoder(nn.Module):
    """Stacked transformer encoder."""
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, tgt_layer=None, tgt_pos=None, tmp_score=None, imp_pos=None, imp_op=None):
        ffn_weights = None
        if imp_op == 'return':
            imp_weights = []
        for i, layer_module in enumerate(self.layer):
            imp_pos_i = [x for x in imp_pos if x[0] == i] if imp_pos else None
            if imp_op == 'return':
                if tgt_layer == i:
                    hidden_states, ffn_weights, imp_i = layer_module(hidden_states, attention_mask, tgt_pos, tmp_score, imp_pos_i, imp_op)
                else:
                    hidden_states, _, imp_i = layer_module(hidden_states, attention_mask, tgt_pos, None, imp_pos_i, imp_op)
                imp_weights.extend(imp_i)
            else:
                if tgt_layer == i:
                    hidden_states, ffn_weights = layer_module(hidden_states, attention_mask, tgt_pos, tmp_score, imp_pos_i, imp_op)
                else:
                    hidden_states, _ = layer_module(hidden_states, attention_mask, tgt_pos, None, imp_pos_i, imp_op)
        return (hidden_states, ffn_weights, imp_weights) if imp_op == 'return' else (hidden_states, ffn_weights)

# -------------------------------------------------------------------------
# Heads
# -------------------------------------------------------------------------
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        return self.LayerNorm(self.act_fn(self.dense(hidden_states)))


class BertLMPredictionHead(nn.Module):
    """MLM head using shared embeddings."""
    def __init__(self, config, embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
        self.decoder.weight = embedding_weights
        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))

    def forward(self, hidden_states):
        return self.decoder(self.transform(hidden_states)) + self.bias


class SentimentClassifier(nn.Module):
    """Linear sentiment classifier using the [CLS] token."""
    def __init__(self, config, hidden_size, num_labels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)
        self.act_fn = ACT2FN[config.hidden_act]
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, sequence_output):
        cls = self.norm(self.act_fn(sequence_output[:, 0]))
        return self.linear(cls)


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, embedding_weights)

    def forward(self, seq_output):
        return self.predictions(seq_output)

# -------------------------------------------------------------------------
# Core Models
# -------------------------------------------------------------------------
class BertModel(PreTrainedBertModel):
    """Base BERT encoder used by ASCENT."""
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                tgt_pos=None, tgt_layer=None, tmp_score=None, imp_pos=None, imp_op=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        if imp_op == 'return':
            encoded_layers, ffn_weights, imp_weights = self.encoder(
                embedding_output, extended_mask, tgt_layer, tgt_pos, tmp_score, imp_pos, imp_op)
            return encoded_layers, ffn_weights, imp_weights
        else:
            encoded_layers, ffn_weights = self.encoder(
                embedding_output, extended_mask, tgt_layer, tgt_pos, tmp_score, imp_pos, imp_op)
            return encoded_layers, ffn_weights


class BertForMaskedLM(PreTrainedBertModel):
    """ASCENT model: combines MLM and sentiment objectives."""
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.sentiment_classifier = SentimentClassifier(config, config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                tgt_pos=None, tgt_layer=None, tmp_score=None, tgt_label=None,
                imp_pos=None, imp_op=None):
        if tmp_score is not None:
            bsz = tmp_score.shape[0]
            input_ids = input_ids.repeat(bsz, 1)
            token_type_ids = token_type_ids.repeat(bsz, 1)
            attention_mask = attention_mask.repeat(bsz, 1)

        if imp_op == 'return':
            last_hidden, ffn_weights, imp_weights = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tgt_pos=tgt_pos,
                tgt_layer=tgt_layer,
                tmp_score=tmp_score,
                imp_pos=imp_pos,
                imp_op=imp_op
            )
        else:
            last_hidden, ffn_weights = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tgt_pos=tgt_pos,
                tgt_layer=tgt_layer,
                tmp_score=tmp_score,
                imp_pos=imp_pos,
                imp_op=imp_op
            )

        if tgt_pos is not None:
            last_hidden_mlm = last_hidden[:, tgt_pos, :]
            tgt_logits = self.cls(last_hidden_mlm)
            tgt_prob = F.softmax(tgt_logits, dim=1)
        else:
            tgt_logits, tgt_prob, ffn_weights = None, None, None

        sentiment_logits = self.sentiment_classifier(last_hidden)

        if imp_op == 'return':
            return imp_weights
        else:
            if tmp_score is None:
                return last_hidden, tgt_prob, tgt_logits, sentiment_logits, ffn_weights
            else:
                gradient = torch.autograd.grad(torch.unbind(tgt_prob[:, tgt_label]), tmp_score)
                return last_hidden, tgt_prob, tgt_logits, sentiment_logits, gradient[0]
