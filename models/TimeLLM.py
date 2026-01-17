from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, TokenEmbedding, ReplicationPad1d
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FrequencyAwarePatchBlock(nn.Module):
    """
    Learnable Frequency-Aware Multi-Scale Patching Block.
    
    Features:
    - FFT-based frequency analysis
    - Direct MLP mapping from frequency to scale weights
    - Residual connections for gradient flow
    """
    
    def __init__(self, configs, candidate_patch_lens=None, dropout=0.1):
        super(FrequencyAwarePatchBlock, self).__init__()
        
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        
        # Auto-compute candidate patch lengths if not provided
        if candidate_patch_lens is None:
            base = max(4, self.seq_len // 24)
            candidate_patch_lens = [base, base * 2, base * 4, min(base * 8, self.seq_len // 2)]
            candidate_patch_lens = sorted(list(set(candidate_patch_lens)))
        
        self.candidate_patch_lens = candidate_patch_lens
        self.n_scales = len(candidate_patch_lens)
        self.strides = [max(1, pl // 2) for pl in candidate_patch_lens]
        self.patch_nums_list = [
            int((self.seq_len - pl) / st + 2)
            for pl, st in zip(self.candidate_patch_lens, self.strides)
        ]
        self.total_patch_nums = sum(self.patch_nums_list)
        
        # Frequency encoder: FFT magnitude -> d_model
        self.freq_dim = self.seq_len // 2 + 1
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.freq_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        
        # Direct scale predictor: d_model -> n_scales
        self.scale_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.n_scales),
        )
        
        # Learnable scale bias
        self.scale_bias = nn.Parameter(torch.zeros(self.n_scales))
        
        # Patch embeddings per scale
        self.padding_layers = nn.ModuleList([
            ReplicationPad1d((0, st)) for st in self.strides
        ])
        self.patch_embeddings = nn.ModuleList([
            TokenEmbedding(pl, self.d_model) for pl in self.candidate_patch_lens
        ])
        
        # Residual projection per scale
        self.residual_proj = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model) for _ in range(self.n_scales)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"[FrequencyAwarePatchBlock] patch_lens={self.candidate_patch_lens}, total_patches={self.total_patch_nums}")
    
    def forward(self, x):
        """
        Args:
            x: (B, N, T) - batch, variables, time
        Returns:
            enc_out: (B*N, total_patches, d_model)
            n_vars: number of variables
            scale_weights: (B*N, n_scales)
        """
        B, N, T = x.shape
        x_flat = x.reshape(B * N, T)
        
        # FFT -> frequency features
        x_fft = torch.fft.rfft(x_flat.float(), dim=-1)
        magnitude = torch.log1p(torch.abs(x_fft))
        magnitude = magnitude.to(next(self.freq_encoder.parameters()).dtype)
        freq_repr = self.freq_encoder(magnitude)  # (B*N, d_model)
        
        # Direct scale prediction
        scale_logits = self.scale_predictor(freq_repr) + self.scale_bias  # (B*N, n_scales)
        scale_weights = F.softmax(scale_logits, dim=-1)
        
        # Multi-scale patching with learned weights
        all_patches = []
        target_dtype = next(self.patch_embeddings[0].parameters()).dtype
        
        for i in range(self.n_scales):
            # Create patches
            x_pad = self.padding_layers[i](x)
            patches = x_pad.unfold(dimension=-1, size=self.candidate_patch_lens[i], step=self.strides[i])
            patches = patches.reshape(B * N, patches.shape[2], patches.shape[3])
            
            # Cast to model dtype for mixed precision compatibility
            patches = patches.to(target_dtype)
            
            # Embed + residual
            patches = self.patch_embeddings[i](patches)
            patches = patches + self.residual_proj[i](patches)  # Residual path
            
            # Weight by scale importance
            weight = scale_weights[:, i:i+1].unsqueeze(-1)
            all_patches.append(patches * weight)
        
        enc_out = self.dropout(torch.cat(all_patches, dim=1))
        return enc_out, N, scale_weights


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


class Model(nn.Module):

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

        # ===== PATCHING MODE SELECTION =====
        # Options: 'frequency_aware' (learnable), 'multi_scale' (fixed), 'single' (original)
        self.patching_mode = getattr(configs, 'patching_mode', 'frequency_aware')
        
        # Get custom candidate patch lengths if provided
        self.candidate_patch_lens = getattr(configs, 'candidate_patch_lens', None)
        
        if self.patching_mode == 'frequency_aware':
            # ===== LEARNABLE FREQUENCY-AWARE PATCHING =====
            self.freq_patch_block = FrequencyAwarePatchBlock(
                configs, 
                candidate_patch_lens=self.candidate_patch_lens,
                dropout=configs.dropout
            )
            self.total_patch_nums = self.freq_patch_block.total_patch_nums
            print(f"[Frequency-Aware] Using learnable multi-scale patching")
            
        elif self.patching_mode == 'multi_scale':
            # ===== FIXED MULTI-SCALE PATCHING (Legacy) =====
            # FFT-optimized patches for weather (seq_len=96)
            self.patch_lens = [8, 16, 32]   # Short, Medium, Long
            self.strides = [4, 8, 16]       # 50% overlap
            
            self.patch_nums_list = [
                int((configs.seq_len - pl) / st + 2) 
                for pl, st in zip(self.patch_lens, self.strides)
            ]
            self.total_patch_nums = sum(self.patch_nums_list)
            print(f"[Multi-Scale] patches={self.patch_lens}, total={self.total_patch_nums}")
        else:  # 'single' mode - original single-scale patching
            self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
            self.total_patch_nums = self.patch_nums
            print(f"[Single-Scale] patch_len={self.patch_len}, stride={self.stride}, patches={self.patch_nums}")

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
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
            except EnvironmentError:  # downloads model from HF is not already done
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
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
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
            except EnvironmentError:  # downloads model from HF is not already done
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
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
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
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # ===== PATCH EMBEDDINGS =====
        if self.patching_mode == 'frequency_aware':
            # Patch embeddings are inside FrequencyAwarePatchBlock
            pass
        elif self.patching_mode == 'multi_scale':
            self.patch_embeddings = nn.ModuleList([
                PatchEmbedding(configs.d_model, pl, st, configs.dropout)
                for pl, st in zip(self.patch_lens, self.strides)
            ])
        else:  # 'single' mode
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.head_nf = self.d_ff * self.total_patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, dynamic_prompts=None):
        """
        Forward pass with optional dynamic prompts.
        
        Args:
            x_enc: Input time series (batch, seq_len, features)
            x_mark_enc: Time encoding for encoder
            x_dec: Decoder input
            x_mark_dec: Time encoding for decoder
            mask: Optional mask
            dynamic_prompts: List of strings, one prompt per sample in batch (optional)
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, dynamic_prompts)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dynamic_prompts=None):
        """
        Forecasting with optional dynamic prompts.
        
        Args:
            dynamic_prompts: List of prompt strings, one per sample (e.g., expert analysis).
        """

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
            # Get batch index (accounting for feature expansion)
            batch_idx = b // N
            
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            trend_str = 'upward' if trends[b] > 0 else 'downward'
            
            # Build the prompt - with or without dynamic expert advice
            if dynamic_prompts is not None and batch_idx < len(dynamic_prompts):
                # Use dynamic prompt (e.g., ChatGPT-generated expert analysis)
                expert_advice = dynamic_prompts[batch_idx]
                prompt_ = (
                    f"<|start_prompt|>"
                    f"Dataset: {self.description} "
                    f"Expert Analysis: {expert_advice} "
                    f"Task: Forecast the next {self.pred_len} {'step' if self.pred_len == 1 else 'steps'} "
                    f"based on the previous {self.seq_len} steps. "
                    f"Current statistics - Min: {min_values_str}, Max: {max_values_str}, "
                    f"Median: {median_values_str}, Trend: {trend_str}, "
                    f"Key lags: {lags_values_str}"
                    f"<|end_prompt|>"
                )
            else:
                # Fallback to static prompt
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description}"
                    f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {trend_str}, "
                    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
                )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        
        # ===== PATCHING =====
        if self.patching_mode == 'frequency_aware':
            # Learnable frequency-aware multi-scale patching
            enc_out, n_vars, scale_weights = self.freq_patch_block(x_enc)
            # Store scale weights for analysis (optional)
            self._last_scale_weights = scale_weights
        elif self.patching_mode == 'multi_scale':
            # Fixed multi-scale patching
            enc_outs = []
            for patch_emb in self.patch_embeddings:
                enc_out_scale, n_vars = patch_emb(x_enc)
                enc_outs.append(enc_out_scale)
            enc_out = torch.cat(enc_outs, dim=1)  # Concatenate all scales
        else:  # 'single' mode
            enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

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
    
    def get_scale_analysis(self):
        """Get analysis of learned scale weights."""
        if self.patching_mode != 'frequency_aware':
            return None
        if hasattr(self, '_last_scale_weights') and self._last_scale_weights is not None:
            weights = self._last_scale_weights.mean(dim=0).float().cpu().numpy()
            return {'weights': weights, 'patch_lens': self.freq_patch_block.candidate_patch_lens}
        return None
    
    def get_learnable_patch_params(self):
        """Get learnable patch parameters."""
        if self.patching_mode != 'frequency_aware':
            return None
        return {
            'scale_bias': self.freq_patch_block.scale_bias.detach().float().cpu(),
            'patch_lens': self.freq_patch_block.candidate_patch_lens,
        }
    
    def print_patch_info(self, epoch=None):
        """Print learned patch length information."""
        if self.patching_mode != 'frequency_aware':
            return
        
        with torch.no_grad():
            patch_lens = self.freq_patch_block.candidate_patch_lens
            scale_bias = self.freq_patch_block.scale_bias.detach().float().cpu().numpy()
            bias_weights = F.softmax(self.freq_patch_block.scale_bias.float(), dim=0).cpu().numpy()
            
            epoch_str = f"Epoch {epoch}" if epoch is not None else "Current"
            print(f"\n[Patch Info] {epoch_str}")
            print(f"  Scale Bias: {scale_bias}")
            print(f"  Weights:    {bias_weights}")
            print(f"  Patch Lens: {patch_lens}")
            
            if hasattr(self, '_last_scale_weights') and self._last_scale_weights is not None:
                last_weights = self._last_scale_weights.mean(dim=0).float().cpu().numpy()
                print(f"  Last Batch: {last_weights}")


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
