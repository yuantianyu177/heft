import torch
import importlib
import uuid

machine_id = uuid.getnode()
if machine_id == 255503834248574 or machine_id == 255503834248024:
    MODEL_CACHE_DIR = "/share/yty/pretrained_models"
elif machine_id == 66988331533420:
    MODEL_CACHE_DIR = "/data/yty/pretrained_models"
else:
    raise ValueError(f"Unsupported machine id: {machine_id}")

# Pipeline parameters
MODEL_PATH = "model_path"
PIPELINE = "pipeline"
PIPELINE_MODULE = "pipeline_module"
ATTENTION_HOOK = "attention_hook"
ATTENTION_HOOK_MODULE = "attention_hook_module"
HOOKED_MODULE = "hooked_module"
PIPELINE_KWARGS = "pipeline_kwargs"
GENERATION_KWARGS = "generation_kwargs"
ATTENTION_HOOK_KWARGS = "attention_hook_kwargs"
CHUNK_SIZE = "chunk_size"

# Generation parameters
PROMPT = "prompt"
FRAMES = "num_frames"
STEPS = "num_inference_steps"
SAVE_STEPS = "save_steps"
SAVE_LAYERS = "save_layers"
SEED = "seed"
POOLING = "pooling"
MODEL = "model"
TASK_NAME = "task_name"
SAVE_ATTENTION = "save_attention_weights"
SAVE_QK = "save_qk_descriptor"
SAVE_HIDDEN_STATES = "save_hidden_states"
FRAME_PATH = "frames"
VIDEO_PATH = "video"
START_STEP = "start_step"
VAE_TEMPORAL_SCALE = 4
RESOLUTION = "resolution"

# Pipeline configuration
PIPELINE_CONFIG = {
    "wan2.1": {
        MODEL_PATH: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        PIPELINE: "WanPipeline",
        PIPELINE_MODULE: "diffusers",
        ATTENTION_HOOK: "WanAttentionHook",
        ATTENTION_HOOK_MODULE: "diffusion_tracker.attention_hook.wan_attention_hook",
        HOOKED_MODULE: "transformer",
        PIPELINE_KWARGS: {
            "torch_dtype": torch.bfloat16,
        },
        ATTENTION_HOOK_KWARGS: {
            "debug": True,
        },
        GENERATION_KWARGS: {
            "output_type": "pil",
        }
    },
    "cosmos2": {
        MODEL_PATH: "nvidia/Cosmos-Predict2-2B-Video2World",
        PIPELINE: "Cosmos2VideoToWorldPipeline",
        PIPELINE_MODULE: "diffusers",
        ATTENTION_HOOK: "CosmosAttentionHook",
        ATTENTION_HOOK_MODULE: "diffusion_tracker.attention_hook.cosmos_attention_hook",
        HOOKED_MODULE: "transformer",
        PIPELINE_KWARGS: {
            "torch_dtype": torch.bfloat16,
        },
        ATTENTION_HOOK_KWARGS: {
            "debug": True,
        }
    },
    "cogvideox": {
        MODEL_PATH: "THUDM/CogVideoX-2b",
        PIPELINE: "CogVideoXPipeline",
        PIPELINE_MODULE: "diffusers",
        ATTENTION_HOOK: "CogVideoXAttentionHook",
        ATTENTION_HOOK_MODULE: "diffusion_tracker.attention_hook.cogvideox_attention_hook",
        HOOKED_MODULE: "transformer",
        PIPELINE_KWARGS: {
            "torch_dtype": torch.bfloat16,
        },
        ATTENTION_HOOK_KWARGS: {
            "debug": True,
        }
    },
}

def import_pipeline(pipeline_name: str):
    if pipeline_name not in PIPELINE_CONFIG:
        raise ValueError(f"Unsupported pipeline: {pipeline_name}. Supported pipelines: {list(PIPELINE_CONFIG.keys())}")
    
    config = PIPELINE_CONFIG[pipeline_name]
    
    module = importlib.import_module(config[PIPELINE_MODULE])
    pipeline = getattr(module, config[PIPELINE])
    
    return pipeline, config

def import_attention_hook(pipeline_name: str):
    if pipeline_name not in PIPELINE_CONFIG:
        raise ValueError(f"Unsupported pipeline: {pipeline_name}. Supported pipelines: {list(PIPELINE_CONFIG.keys())}")
    
    config = PIPELINE_CONFIG[pipeline_name]
    
    hook_module = importlib.import_module(config[ATTENTION_HOOK_MODULE])
    attention_hook = getattr(hook_module, config[ATTENTION_HOOK])
    
    return attention_hook