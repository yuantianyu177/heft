import torch
import json
import os
from diffusers.utils import export_to_video
from .utils.generation import *
from .utils.io import load_video, save_video


def single_generation(config: dict, base_save_dir: str):
    # Load model
    pipeline, pipeline_config = import_pipeline(config[MODEL])
    pipe = pipeline.from_pretrained(
        pipeline_config[MODEL_PATH],
        cache_dir=MODEL_CACHE_DIR,
        **pipeline_config.get(PIPELINE_KWARGS, {}),
    )
    pipe.safety_checker = None
    pipe.enable_model_cpu_offload()
    # pipe.enable_vae_tiling()
    
    task_name = config.get(TASK_NAME, f"task_{config.get(SEED, 42)}")
    save_dir = os.path.join(base_save_dir, task_name)
    frame_dir = os.path.join(save_dir, FRAME_PATH)
    video_dir = os.path.join(save_dir, VIDEO_PATH)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    # Save config
    with open(f"{save_dir}/generation_params.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        
    # Load input video if provided
    if config[VIDEO_PATH]:
        input_video = load_video(config[VIDEO_PATH], normalize=True, output_size=config[RESOLUTION])
        if config[FRAMES] != None:
            total_frames = config[FRAMES]
            chunk_size = config[FRAMES]
        else:
            total_frames = input_video.shape[0]
            chunk_size = config[CHUNK_SIZE]
            total_frames = (total_frames // chunk_size) * chunk_size

        input_video = input_video[:total_frames]
        save_video(input_video*255, os.path.join(video_dir, "original_video.mp4"), fps=10)
        num_chunks = total_frames // chunk_size 
    else:
        input_video = None
        total_frames = config[FRAMES]
        chunk_size = total_frames
        num_chunks = 1
    
    # Process each chunk
    all_frames = []
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_size
        end_frame = min(start_frame + chunk_size, total_frames)
        num_frames_chunk = end_frame - start_frame
             
        # Get chunk of input video
        if input_video is not None:
            input_video_chunk = input_video[start_frame:end_frame]
        else:
            input_video_chunk = None
        
        # Create attention hook with chunk_id
        attention_hook_cls = import_attention_hook(config[MODEL])
        hooked_module = getattr(pipe, pipeline_config[HOOKED_MODULE])
        attention_hook_instance = attention_hook_cls(
            save_dir=save_dir,
            model=hooked_module,
            save_steps=config[SAVE_STEPS],
            save_layers=config[SAVE_LAYERS],
            pooling=config[POOLING],
            save_attention_weights=config[SAVE_ATTENTION],
            save_qk_descriptor=config[SAVE_QK],
            save_hidden_states=config[SAVE_HIDDEN_STATES],
            start_step=config[START_STEP],
            chunk_id=chunk_idx,
            **pipeline_config.get(ATTENTION_HOOK_KWARGS, {}),
        )
        
        # Generate video for this chunk
        generator = torch.manual_seed(config[SEED])
        frames = pipe(
            prompt=config[PROMPT],
            num_frames=num_frames_chunk,
            num_inference_steps=config[STEPS],
            generator=generator,
            **pipeline_config.get(GENERATION_KWARGS, {}),
            input_video=input_video_chunk,
            start_step=config.get(START_STEP, None),
        ).frames[0]
        
        all_frames.extend(frames)
        
        # Export frames for this chunk
        for i, frame in enumerate(frames):
            frame_idx = start_frame + i
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:03d}.png")
            frame.save(frame_path)
        
        # Clean up hooks for this chunk
        attention_hook_instance.remove_hooks()
        del attention_hook_instance
        torch.cuda.empty_cache()

    # Export complete video
    export_to_video(all_frames, os.path.join(video_dir, "video.mp4"), fps=10)