import os
import argparse
from multiprocessing import Process, Semaphore
from heft.video_generator import single_generation
from heft.utils.generation import *


def worker(gpu_id, config, save_dir, sem):
    with sem:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print(f"Process using GPU {gpu_id}")
        single_generation(config, save_dir)

def generate_configs(args) -> list[dict]:
    assert (args.prompts_file is None) ^ (args.video_file is None), \
        "Either prompts_file or video_file must be provided"
    assert not (args.prompts_file is not None and args.num_frames is None), \
        "In generation mode, num_frames must be provided"
    use_prompt = True if args.prompts_file is not None else False
    task = []
    with open(args.prompts_file if use_prompt else args.video_file, 'r', encoding='utf-8') as f:
        for _, line in enumerate(f, 1):
            line = line.strip()
            if line:
                task.append(line)
    
    configs = []
    for i, info in enumerate(task):     
        config = {
        PROMPT: info if use_prompt else "",
        FRAMES: args.num_frames,
        STEPS: args.num_inference_steps,
        SAVE_STEPS: args.save_steps,
        SAVE_LAYERS: args.save_layers,
        SEED: args.seed,
        POOLING: args.pooling,
        MODEL: args.model,
        START_STEP: args.start_step,
        VIDEO_PATH: None if use_prompt else info,
        SAVE_ATTENTION: args.save_attention,
        SAVE_QK: args.save_qk,
        SAVE_HIDDEN_STATES: args.save_hidden_states,
        TASK_NAME: f"task_{i:04d}",
        CHUNK_SIZE: args.chunk_size,
        RESOLUTION: args.resolution
    }
        configs.append(config)
    
    return configs

def main(args):
    configs = generate_configs(args)
    gpu_sems = {gpu: Semaphore(1) for gpu in args.gpu}
    processes = []
    for i, config in enumerate(configs):
        gpu_id = args.gpu[i % len(args.gpu)]
        sem = gpu_sems[gpu_id]
        p = Process(target=worker, args=(gpu_id, config, args.save_dir, sem))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU generation script")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--model", type=str, default="wan2.1")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=[480, 832])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--save_steps", nargs='+', type=int, default=None)
    parser.add_argument("--save_layers", nargs='+', type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", action="store_true")
    parser.add_argument("--save_attention", action="store_true")
    parser.add_argument("--save_qk", action="store_true")
    parser.add_argument("--save_hidden_states", action="store_true")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=25)
    parser.add_argument("--gpu", nargs='+', type=int, default=[0])
    parser.add_argument("--save_dir", type=str, default="./output") 
    args = parser.parse_args()
    main(args)