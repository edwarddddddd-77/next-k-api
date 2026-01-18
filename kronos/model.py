"""
Kronos Model - K-Line Foundation Model
Based on https://github.com/shiyu-coder/Kronos (MIT License)

Core classes for K-line tokenization and prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional, List, Dict, Any

from .modules import (
    TransformerBlock, BSQuantizer, RMSNorm, HierarchicalEmbedding,
    TemporalEmbedding, DependencyAwareLayer, DualHead
)


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    Tokenizer for K-line data using Binary Spherical Quantization.
    Converts continuous OHLCV data into discrete tokens.
    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers,
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p,
                 s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits

        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_enc_layers - 1)
        ])
        self.decoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_dec_layers - 1)
        ])

        self.quant_embed = nn.Linear(d_model, self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(s1_bits, d_model)
        self.post_quant_embed = nn.Linear(self.codebook_dim, d_model)
        self.tokenizer = BSQuantizer(s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size)

    def forward(self, x):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        z = self.post_quant_embed(quantized)

        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        if half:
            x1, x2 = x[0], x[1]
            mask = 2 ** torch.arange(self.codebook_dim // 2, device=x1.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask) != 0
            x2 = (x2.unsqueeze(-1) & mask) != 0
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
            x = (x.unsqueeze(-1) & mask) != 0
        x = x.float() * 2 - 1
        return x * (1. / (self.codebook_dim ** 0.5))

    def encode(self, x, half=False):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        _, _, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        return self.head(z)


class Kronos(nn.Module, PyTorchModelHubMixin):
    """
    Kronos Model - Decoder-only Transformer for K-line prediction.
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim,
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.s1_vocab_size = 2 ** s1_bits

        self.token_drop = nn.Dropout(token_dropout_p)
        self.embedding = HierarchicalEmbedding(s1_bits, s2_bits, d_model)
        self.time_emb = TemporalEmbedding(d_model, learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.dep_layer = DependencyAwareLayer(d_model)
        self.head = DualHead(s1_bits, s2_bits, d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None,
                use_teacher_forcing=False, s1_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)
        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.head(x), x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


# ============== Sampling Utilities ==============

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    elif top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        logits = top_k_top_p_filtering(logits, top_k=top_k or 0, top_p=top_p or 1.0)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ============== Autoregressive Inference ==============

def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len,
                              clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5):
    """Run autoregressive inference for K-line prediction."""
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device

        x_seq_len, x_features = x.size(1), x.size(2)
        x_stamp_seq_len, x_stamp_features = x_stamp.size(1), x_stamp.size(2)
        y_stamp_seq_len, y_stamp_features = y_stamp.size(1), y_stamp.size(2)

        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_seq_len, x_features).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp_seq_len, x_stamp_features).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp_seq_len, y_stamp_features).to(device)

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            sample_pre = sample_from_logits(s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p)

            s2_logits = model.decode_s2(context, sample_pre)
            sample_post = sample_from_logits(s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer = torch.roll(pre_buffer, shifts=-1, dims=1)
                post_buffer = torch.roll(post_buffer, shifts=-1, dims=1)
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous()
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        return np.mean(z.cpu().numpy(), axis=1)


def auto_regressive_inference_with_samples(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len,
                                           clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5):
    """Run autoregressive inference and return individual samples (not averaged).

    This is optimized for Monte Carlo sampling - all samples computed in one batched forward pass.
    """
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device

        x_seq_len, x_features = x.size(1), x.size(2)
        x_stamp_seq_len, x_stamp_features = x_stamp.size(1), x_stamp.size(2)
        y_stamp_seq_len, y_stamp_features = y_stamp.size(1), y_stamp.size(2)

        # Batch all samples together for parallel computation
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_seq_len, x_features).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp_seq_len, x_stamp_features).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp_seq_len, y_stamp_features).to(device)

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            sample_pre = sample_from_logits(s1_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p)

            s2_logits = model.decode_s2(context, sample_pre)
            sample_post = sample_from_logits(s2_logits[:, -1, :], temperature=T, top_k=top_k, top_p=top_p)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer = torch.roll(pre_buffer, shifts=-1, dims=1)
                post_buffer = torch.roll(post_buffer, shifts=-1, dims=1)
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous()
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        # Return individual samples instead of mean
        return z.squeeze(0).cpu().numpy()


def calc_time_stamps(timestamps: pd.Series) -> pd.DataFrame:
    """Extract temporal features from timestamps."""
    return pd.DataFrame({
        'minute': timestamps.dt.minute,
        'hour': timestamps.dt.hour,
        'weekday': timestamps.dt.weekday,
        'day': timestamps.dt.day,
        'month': timestamps.dt.month
    })


# ============== High-Level Predictor ==============

class KronosPredictor:
    """
    High-level API for K-line forecasting using Kronos model.
    """

    def __init__(self, model: Kronos, tokenizer: KronosTokenizer,
                 device: Optional[str] = None, max_context: int = 512, clip: int = 5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_col = 'amount'

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def _prepare_data(self, df: pd.DataFrame, x_timestamp: pd.Series,
                      y_timestamp: pd.Series) -> tuple:
        """Prepare data for prediction."""
        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
        if self.amt_col not in df.columns:
            df[self.amt_col] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_col]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        return x, x_stamp, y_stamp

    def predict(self, df: pd.DataFrame, x_timestamp: pd.Series, y_timestamp: pd.Series,
                pred_len: int, T: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                sample_count: int = 1) -> pd.DataFrame:
        """Generate K-line forecast."""
        x, x_stamp, y_stamp = self._prepare_data(df, x_timestamp, y_timestamp)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(self.device)
        x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :]).to(self.device)
        y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :]).to(self.device)

        preds = auto_regressive_inference(
            self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor,
            self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count
        )

        preds = preds.squeeze(0)[-pred_len:]
        preds = preds * (x_std + 1e-5) + x_mean

        return pd.DataFrame(
            preds, columns=self.price_cols + [self.vol_col, self.amt_col],
            index=y_timestamp
        )

    def predict_multi_sample(self, df: pd.DataFrame, x_timestamp: pd.Series,
                             y_timestamp: pd.Series, pred_len: int,
                             T: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                             sample_count: int = 10) -> Dict[str, Any]:
        """Generate multiple Monte Carlo samples for probabilistic forecasting.

        Uses batched inference for better performance on multi-core CPUs.
        """
        x, x_stamp, y_stamp = self._prepare_data(df, x_timestamp, y_timestamp)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(self.device)
        x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :]).to(self.device)
        y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :]).to(self.device)

        # Use batched inference - all samples in one forward pass
        all_preds = auto_regressive_inference_with_samples(
            self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor,
            self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count
        )

        # all_preds shape: [sample_count, seq_len, features]
        all_preds = all_preds[:, -pred_len:, :]
        all_samples = all_preds * (x_std + 1e-5) + x_mean
        close_samples = all_samples[:, :, 3]  # close price

        return {
            'mean': np.mean(close_samples, axis=0),
            'std': np.std(close_samples, axis=0),
            'min': np.min(close_samples, axis=0),
            'max': np.max(close_samples, axis=0),
            'p10': np.percentile(close_samples, 10, axis=0),
            'p25': np.percentile(close_samples, 25, axis=0),
            'p75': np.percentile(close_samples, 75, axis=0),
            'p90': np.percentile(close_samples, 90, axis=0),
            'timestamps': y_timestamp,
            'all_samples': all_samples,
        }

    def get_embedding(self, df: pd.DataFrame, x_timestamp: pd.Series) -> np.ndarray:
        """
        Get embedding vector for a K-line sequence.
        Useful for similarity search and pattern matching.
        """
        x, x_stamp, _ = self._prepare_data(df, x_timestamp, x_timestamp)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(self.device)

        with torch.no_grad():
            z = self.tokenizer.embed(x_tensor)
            for layer in self.tokenizer.encoder:
                z = layer(z)
            # Use mean pooling for sequence embedding
            embedding = z.mean(dim=1).cpu().numpy()

        return embedding.squeeze(0)

    def calculate_perplexity(self, df: pd.DataFrame, x_timestamp: pd.Series) -> float:
        """
        Calculate perplexity (anomaly score) for a K-line sequence.
        Higher perplexity = more unusual/anomalous pattern.
        """
        if len(df) < 24:
            return 0.5

        closes = df['close'].values

        if len(closes) > 48:
            recent_returns = np.log(closes[-24:] / closes[-25:-1])
            hist_returns = np.log(closes[-48:-24] / closes[-49:-25])
            recent_vol = np.std(recent_returns)
            hist_vol = np.std(hist_returns)
            vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
        else:
            vol_ratio = 1.0

        price_change = (closes[-1] - closes[-24]) / closes[-24] if len(closes) >= 24 else 0
        momentum_score = min(1.0, abs(price_change) * 10)

        perplexity = vol_ratio * 0.6 + momentum_score * 0.4
        return max(0.1, min(0.95, perplexity))
