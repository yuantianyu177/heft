import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .base_attention_hook import AttentionHook
from einops import rearrange
from ..utils.misc import printi, printw, printe
from ..utils.generation import *

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)

    
class CogVideoXProcessorAdapter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._last_attention_weights = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXProcessorAdapter requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

        self.save_steps = kwargs.get(SAVE_STEPS, None)
        self.save_layers = kwargs.get(SAVE_LAYERS, None)
        self.layer_idx = kwargs.get("layer_idx", 0)
        self.use_pooling = kwargs.get(POOLING, True)
        self.save_attention_weights = kwargs.get(SAVE_ATTENTION, False)
        self.save_qk_descriptor = kwargs.get(SAVE_QK, False)
        self.save_intermediate_hidden_states = kwargs.get(SAVE_HIDDEN_STATES, False)
        self.current_step = kwargs.get(START_STEP, 0)
        self.chunk_id = kwargs.get("chunk_id", None)
        self.pooling = nn.AdaptiveAvgPool2d((256, 256))

    def get_attention_weights(self, query, key):
        seq_len = query.size(2)
        num_heads = query.size(1)
        identity_value_head = torch.eye(seq_len, device=query.device, dtype=query.dtype)
        identity_value_head = identity_value_head.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]
        self._last_attention_weights = torch.zeros(
            (1, num_heads, seq_len, seq_len),
            device=query.device,
            dtype=query.dtype
        )
        for head_idx in range(num_heads):
            query_head = query[:, head_idx:head_idx+1, :, :]
            key_head = key[:, head_idx:head_idx+1, :, :]
            self._last_attention_weights[:, head_idx:head_idx+1, :, :] = F.scaled_dot_product_attention(
                query_head, key_head, identity_value_head, dropout_p=0.0, is_causal=False
            )
        self._last_attention_weights = rearrange(self._last_attention_weights, "b head h w -> (b head) () h w")
        if self.use_pooling:
            self._last_attention_weights = self.pooling(self._last_attention_weights)
        self._last_attention_weights = rearrange(self._last_attention_weights, "(b head) () h w -> b head h w", b=1)
        del identity_value_head, query_head, key_head

    def get_qk_descriptor(self, query, key):
        self._last_query = query # (b, h, n, c)
        self._last_key = key

    def get_intermediate_hidden_states(self, hidden_states):
        self._last_hidden_states = hidden_states

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # -----------------------------------------------------------------------------
        self.record = False
        if self.save_steps is None or self.current_step in self.save_steps:
            if self.save_layers is None or self.layer_idx in self.save_layers:
                self.record = True
                self.record_step = self.current_step
                if self.save_attention_weights:
                    self.get_attention_weights(query[1:, :, text_seq_length:], key[1:, :, text_seq_length:])         
                if self.save_qk_descriptor:
                    self.get_qk_descriptor(query[1:, :, text_seq_length:], key[1:, :, text_seq_length:])
                if self.save_intermediate_hidden_states:
                    self.get_intermediate_hidden_states(hidden_states[1:, :, text_seq_length:])
        self.current_step += 1
        # -----------------------------------------------------------------------------

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogVideoXAttentionHook(AttentionHook):
    def __init__(self, **kwargs):
        self.adapter_name = "CogVideoXProcessorAdapter"
        super().__init__(**kwargs)     
        
    def _replace_attn_layer(self, model: nn.Module):
        layer_idx = 0
        for name, module in model.named_modules():
            if hasattr(module, 'attn1'):
                new_processor = self._get_adapter(layer_idx=layer_idx)
                new_processor.eval()
                module.attn1.processor = new_processor
                if self.debug:
                    printi(f"layer {layer_idx} replace {name}.attn1.processor with {new_processor.__class__.__name__}")
                layer_idx += 1