import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
from abc import ABC, abstractmethod
import importlib
from ..utils.misc import printi, printw, printe

class AttentionHook(ABC):   
    def __init__(self, **kwargs):
        self.save_dir = kwargs.get("save_dir", "./")
        self.chunk_id = kwargs.get("chunk_id", None)
        self.attention_dir = kwargs.get("attention_dir", os.path.join(self.save_dir, "attention_maps"))
        self.query_dir = kwargs.get("query_dir", os.path.join(self.save_dir, "query"))
        self.key_dir = kwargs.get("key_dir", os.path.join(self.save_dir, "key"))
        self.hidden_states_dir = kwargs.get("hidden_states_dir", os.path.join(self.save_dir, "hidden_states"))
        self.save_attention_weights = kwargs.get("save_attention_weights", False)
        self.save_qk_descriptor = kwargs.get("save_qk_descriptor", False)
        self.save_hidden_states = kwargs.get("save_hidden_states", False)
        self.save_steps = kwargs.get("save_steps", None)
        self.save_layers = kwargs.get("save_layers", None)
        self.pooling = kwargs.get("pooling", True)
        self.debug = kwargs.get("debug", False)
        self.output_type = kwargs.get("output_type", "pt")
        self.start_step = kwargs.get("start_step", 0)
        self.high_freq_ratio = kwargs.get("high_freq_ratio", 1.0)
        self.hooks = []

        self._replace_attn_layer(kwargs.get("model", None))
        self._register_hooks(kwargs.get("model", None))
        
    def _create_attention_hook(self):
        def hook(module, input, output):
            try:
                if module.record:
                    if self.save_attention_weights:
                        attention_weights = self._extract_attention_weights(module, input, output)
                        for i in range(attention_weights.shape[1]):
                            head_tensor = attention_weights[:, i, :, :].clone()
                            self._record_tensor(head_tensor, module.record_step, module.layer_idx, self.attention_dir, head=i, filename_pattern="head_{head:02d}.{output_type}", desc="attention map")
                    if self.save_qk_descriptor:
                        query, key = self._extract_qk_descriptor(module, input, output)
                        query_channels = query.shape[-1]
                        key_channels = key.shape[-1]
                        query = query[:, :, :, :int(query_channels * self.high_freq_ratio)]
                        key = key[:, :, :, :int(key_channels * self.high_freq_ratio)]
                        self._record_tensor(query, module.record_step, module.layer_idx, self.query_dir, desc="query")
                        self._record_tensor(key, module.record_step, module.layer_idx, self.key_dir, desc="key")
                    if self.save_hidden_states:
                        hidden_states = self._extract_hidden_states(module, input, output)
                        self._record_tensor(hidden_states, module.record_step, module.layer_idx, self.hidden_states_dir, desc="hidden states")  
            except Exception as e:
                printe(f"step {module.record_step}, layer {module.layer_idx} error {e}")
        
        return hook
         
    def _register_hooks(self, model: nn.Module):
        self.hooks = []
        for name, module in model.named_modules():
            if self._is_attention_layer(module):
                hook = module.register_forward_hook(self._create_attention_hook())
                self.hooks.append(hook)
                if self.debug:
                    printi(f"register hook to module: {name}")
        
    def _record_tensor(
        self, 
        tensor: torch.Tensor, 
        step: int, 
        layer: int,
        dir_root: str, 
        head: int=None,
        filename_pattern: str="layer_{layer:02d}.{output_type}", 
        desc: str="tensor"
    ):
        step_dir = os.path.join(dir_root, f"step_{step:03d}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Add chunk_id to filename if it exists
        if head is None:
            if self.chunk_id is not None:
                # Insert chunk_id before the file extension
                # e.g., "layer_{layer:02d}.{output_type}" -> "layer_{layer:02d}_chunk_{chunk_id:03d}.{output_type}"
                base_pattern = filename_pattern.replace(".{output_type}", f"_chunk_{self.chunk_id:03d}.{{output_type}}")
                filepath = os.path.join(step_dir, base_pattern.format(layer=layer, output_type=self.output_type))
            else:
                filepath = os.path.join(step_dir, filename_pattern.format(layer=layer, output_type=self.output_type))
        else:
            layer_dir = os.path.join(step_dir, f"layer_{layer:02d}")
            os.makedirs(layer_dir, exist_ok=True)
            if self.chunk_id is not None:
                # Insert chunk_id before the file extension for head files
                # e.g., "head_{head:02d}.{output_type}" -> "head_{head:02d}_chunk_{chunk_id:03d}.{output_type}"
                base_pattern = filename_pattern.replace(".{output_type}", f"_chunk_{self.chunk_id:03d}.{{output_type}}")
                filepath = os.path.join(layer_dir, base_pattern.format(head=head, output_type=self.output_type))
            else:
                filepath = os.path.join(layer_dir, filename_pattern.format(head=head, output_type=self.output_type))
        
        if self.output_type == "pt":
            torch.save(tensor, filepath)
        elif self.output_type == "npy":
            np.save(filepath, tensor.numpy())
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")
        if self.debug:
            chunk_info = f"chunk {self.chunk_id}, " if self.chunk_id is not None else ""
            printi(f"{chunk_info}step {step}, layer {layer} record {desc}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
   
    def _is_attention_layer(self, module: nn.Module) -> bool:
        if hasattr(module, '__class__') and module.__class__.__name__ == self.adapter_name:
            return True
        return False
    
    def _extract_attention_weights(self, module, input, output) -> Optional[torch.Tensor]:
        if hasattr(module, '_last_attention_weights'):
            attention_weights_cpu = module._last_attention_weights.detach().cpu().to(dtype=torch.float32)
            if attention_weights_cpu is not None:
                module._last_attention_weights = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return attention_weights_cpu
        
        return None

    def _extract_qk_descriptor(self, module, input, output) -> Optional[torch.Tensor]:
        if hasattr(module, '_last_query') and hasattr(module, '_last_key'):
            query_cpu = module._last_query.detach().cpu().to(dtype=torch.float32)
            key_cpu = module._last_key.detach().cpu().to(dtype=torch.float32)
            if query_cpu is not None and key_cpu is not None:
                module._last_query = None
                module._last_key = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return query_cpu, key_cpu
        return None, None
    
    def _extract_hidden_states(self, module, input, output) -> Optional[torch.Tensor]:
        if hasattr(module, '_last_hidden_states'):
            hidden_states_cpu = module._last_hidden_states.detach().cpu().to(dtype=torch.float32)
            if hidden_states_cpu is not None:
                module._last_hidden_states = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return hidden_states_cpu
        return None

    def _get_adapter(self, layer_idx: int) -> nn.Module:
        module = importlib.import_module(self.__class__.__module__)
        cls = getattr(module, self.adapter_name)
        return cls(
            save_steps=self.save_steps, 
            save_layers=self.save_layers,
            layer_idx=layer_idx,
            pooling=self.pooling,
            save_attention_weights=self.save_attention_weights,
            save_qk_descriptor=self.save_qk_descriptor,
            save_hidden_states=self.save_hidden_states,
            start_step=self.start_step,
            chunk_id=self.chunk_id,
        )

    @abstractmethod
    def _replace_attn_layer(self, model: nn.Module):
        pass    