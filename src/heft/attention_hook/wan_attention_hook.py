import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .base_attention_hook import AttentionHook
from einops import rearrange
from ..utils.misc import printi, printw, printe
from ..utils.generation import *


class WanProcessorAdapter(nn.Module):
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        self._last_attention_weights = None
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

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
        
        self.is_cond = True

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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                dtype = torch.float32 if hidden_states.device.type == "mps" else torch.float64
                x_rotated = torch.view_as_complex(hidden_states.to(dtype).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # -----------------------------------------------------------------------------
        self.record = False
        if self.save_steps is None or self.current_step in self.save_steps:
            if self.save_layers is None or self.layer_idx in self.save_layers:
                if self.is_cond:
                    self.record = True
                    self.record_step = self.current_step
                    if self.save_attention_weights:
                        self.get_attention_weights(query, key)         
                    if self.save_qk_descriptor:
                        self.get_qk_descriptor(query, key)
                    if self.save_intermediate_hidden_states:
                        self.get_intermediate_hidden_states(hidden_states)
        # One step contains two forward passes, one conditional, one unconditional
        if self.is_cond:
            self.current_step += 1
        self.is_cond = not self.is_cond
        # -----------------------------------------------------------------------------
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttentionHook(AttentionHook):
    def __init__(self, **kwargs):
        self.adapter_name = "WanProcessorAdapter"
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