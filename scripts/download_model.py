import torch
from diffusers import CogVideoXPipeline, Cosmos2VideoToWorldPipeline, WanPipeline

model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, mirror="https://hf-mirror.com")

# model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# model_id = "THUDM/CogVideoX-2b"
# pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)