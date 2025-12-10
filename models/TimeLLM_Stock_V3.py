"""
TimeLLM Stock V3 Model
Combines:
- Multi-scale patching (5, 10, 20 day windows) for capturing patterns at multiple timescales
- Feature attention for interpretability
- Enhanced trend prompts with momentum analysis
- Dynamic prompts (ChatGPT-generated)
- Support for directional loss training
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FeatureAttention(nn.Module):
    """
    Feature-level attention mechanism to compute importance scores for each input variable.
    This helps identify which features contribute most to the predictions.
    """
    def __init__(self, n_vars, d_model, attention_type='additive'):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.attention_type = attention_type
        
        if attention_type == 'additive':
            # Additive (Bahdanau-style) attention
            self.attention_weights = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )
        elif attention_type == 'multiplicative':
            # Multiplicative (Luong-style) attention with learnable query
            self.query = nn.Parameter(torch.randn(1, 1, d_model))
            self.key_projection = nn.Linear(d_model, d_model)
        elif attention_type == 'self':
            # Self-attention across features
            self.query_proj = nn.Linear(d_model, d_model)
            self.key_proj = nn.Linear(d_model, d_model)
            self.value_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, return_weights=True):
        """
        Args:
            x: (B, N, T, D) where B=batch, N=features, T=time, D=embedding_dim
        Returns:
            weighted_x: (B, N, T, D) - feature-weighted representation
            attention_weights: (B, N) - importance score for each feature
        """
        B, N, T, D = x.shape
        
        # Aggregate temporal dimension to get feature-level representation
        feature_repr = x.mean(dim=2)  # (B, N, D)
        
        if self.attention_type == 'additive':
            scores = self.attention_weights(feature_repr)  # (B, N, 1)
            attention_weights = F.softmax(scores.squeeze(-1), dim=1)  # (B, N)
            
        elif self.attention_type == 'multiplicative':
            queries = self.query.expand(B, -1, -1)  # (B, 1, D)
            keys = self.key_projection(feature_repr)  # (B, N, D)
            scores = torch.bmm(queries, keys.transpose(1, 2)) / sqrt(D)  # (B, 1, N)
            attention_weights = F.softmax(scores.squeeze(1), dim=1)  # (B, N)
            
        elif self.attention_type == 'self':
            Q = self.query_proj(feature_repr)  # (B, N, D)
            K = self.key_proj(feature_repr)  # (B, N, D)
            V = self.value_proj(feature_repr)  # (B, N, D)
            
            scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(D)  # (B, N, N)
            attention_map = F.softmax(scores, dim=-1)  # (B, N, N)
            
            attention_weights = attention_map.sum(dim=1)  # (B, N)
            attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)
        
        # Apply attention weights to scale features
        weighted_x = x * attention_weights.unsqueeze(-1).unsqueeze(-1)
        
        if return_weights:
            return weighted_x, attention_weights
        else:
            return weighted_x


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class Model(nn.Module):
    """
    TimeLLM Stock V3 with:
    - Multi-scale patching
    - Feature attention
    - Enhanced trend prompts
    - Dynamic prompt support
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        
        # ===== MULTI-SCALE PATCHING =====
        # Capture patterns at multiple timescales (5, 10, 20 days)
        self.use_multi_scale = getattr(configs, 'use_multi_scale', True)
        
        if self.use_multi_scale:
            # Short-term (5 days), Medium-term (10 days), Long-term (20 days)
            self.patch_lens = [5, 10, 20]
            self.strides = [3, 5, 10]
            self.patch_nums_list = [
                int((configs.seq_len - pl) / st + 2) 
                for pl, st in zip(self.patch_lens, self.strides)
            ]
            self.total_patch_nums = sum(self.patch_nums_list)
            print(f"[Multi-Scale Patching] Enabled: {self.patch_lens} -> {self.patch_nums_list} patches (total: {self.total_patch_nums})")
        else:
            self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
            self.total_patch_nums = self.patch_nums

        # ===== LLM INITIALIZATION =====
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'VCB stock price prediction with technical indicators.'

        self.dropout = nn.Dropout(configs.dropout)

        # ===== MULTI-SCALE PATCH EMBEDDINGS =====
        if self.use_multi_scale:
            self.patch_embeddings = nn.ModuleList([
                PatchEmbedding(configs.d_model, pl, st, configs.dropout)
                for pl, st in zip(self.patch_lens, self.strides)
            ])
        else:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        if not self.use_multi_scale:
            self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.total_patch_nums
        
        # ===== ENHANCED TREND PROMPTS =====
        self.use_trend_in_prompt = True
        if self.use_trend_in_prompt:
            print(f"[Enhanced Trend Prompt] Enabled")

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        
        # ===== FEATURE ATTENTION =====
        self.use_feature_attention = getattr(configs, 'use_feature_attention', True)
        if self.use_feature_attention:
            attention_type = getattr(configs, 'attention_type', 'additive')
            self.feature_attention = FeatureAttention(configs.enc_in, self.d_ff, attention_type=attention_type)
            print(f"[Feature Attention] Enabled with {attention_type} attention")
        
        # Store attention weights for analysis
        self.last_attention_weights = None
        self.last_feature_importance = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, dynamic_prompts=None, return_attention=False):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, dynamic_prompts)
            if return_attention and self.use_feature_attention:
                return dec_out[:, -self.pred_len:, :], self.last_attention_weights
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dynamic_prompts=None):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            
            # ===== ENHANCED TREND WITH MOMENTUM ANALYSIS =====
            if self.use_trend_in_prompt:
                recent_mean = x_enc[b, -10:].mean().item()
                older_mean = x_enc[b, :10].mean().item()
                trend_strength = abs(recent_mean - older_mean) / (abs(older_mean) + 1e-5)
                recent_slope = x_enc[b, -5:].mean().item() - x_enc[b, -10:-5].mean().item()
                momentum = "accelerating" if recent_slope * trends[b] > 0 else "decelerating"
                trend_desc = f"{'upward' if trends[b] > 0 else 'downward'}, strength {trend_strength:.2f}, {momentum}"
            else:
                trend_desc = f"{'upward' if trends[b] > 0 else 'downward'}"
            
            # ===== BUILD PROMPT WITH DYNAMIC CHATGPT ADVICE =====
            batch_idx = b // N  # Get the actual batch index
            
            # Get dynamic prompt if available
            dynamic_advice = ""
            if dynamic_prompts is not None and batch_idx < len(dynamic_prompts):
                dynamic_advice = f"\nExpert Analysis: {dynamic_prompts[batch_idx]}"
            
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {trend_desc}, "
                f"top 5 lags are : {lags_values_str}"
                f"{dynamic_advice}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        
        # ===== MULTI-SCALE PATCHING =====
        if self.use_multi_scale:
            enc_outs = []
            for patch_emb in self.patch_embeddings:
                # Don't convert to bfloat16 to avoid dtype mismatch
                enc_out_scale, n_vars = patch_emb(x_enc)
                enc_outs.append(enc_out_scale)
            enc_out = torch.cat(enc_outs, dim=1)
        else:
            enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        
        # ===== APPLY FEATURE ATTENTION =====
        if self.use_feature_attention:
            dec_out_patches = dec_out[:, :, :, -self.total_patch_nums:]
            # Permute to (B, N, T, D) for feature attention
            dec_out_patches = dec_out_patches.permute(0, 1, 3, 2).contiguous()
            dec_out_weighted, attention_weights = self.feature_attention(dec_out_patches, return_weights=True)
            # Store for later analysis
            self.last_attention_weights = attention_weights.detach().cpu()
            # Permute back to (B, N, D, T)
            dec_out_weighted = dec_out_weighted.permute(0, 1, 3, 2).contiguous()
            dec_out = self.output_projection(dec_out_weighted)
        else:
            dec_out = self.output_projection(dec_out[:, :, :, -self.total_patch_nums:])
        
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
            
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance scores from the last forward pass.
        
        Args:
            feature_names: List of feature names (optional)
        
        Returns:
            Dictionary mapping feature names/indices to importance scores
        """
        if self.last_attention_weights is None:
            print("Warning: No attention weights available. Run a forward pass first.")
            return None
        
        # Average across batch dimension
        avg_weights = self.last_attention_weights.mean(dim=0).numpy()
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(avg_weights))]
        
        importance_dict = {name: float(weight) for name, weight in zip(feature_names, avg_weights)}
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        self.last_feature_importance = importance_dict
        
        return importance_dict
    
    def reset_attention_weights(self):
        """Reset stored attention weights."""
        self.last_attention_weights = None
        self.last_feature_importance = None

