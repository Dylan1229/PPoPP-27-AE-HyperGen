import os
import torch
from diffsynth import ModelManager, WanVideoPipeline_Pro, save_video
from modelscope import snapshot_download
import torch.distributed as dist

# Download models
# Weights are symlinked in from ~/ckpts; snapshot_download would try to re-fetch
# 65 GB over the network and, worse, may replace the symlink.
if os.environ.get("CS_SKIP_DOWNLOAD", "1") != "1":
    snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir="models/Wan-AI/Wan2.1-T2V-14B")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
            "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
        ],
        "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
)

# The LoRA is CineScale's TRAINED component. Skipping it gives the training-free configuration,
# which is the fair comparison against us: we are training-free by construction, so comparing our
# output against a LoRA-tuned CineScale confounds the method with the tuning.
#
# Both arms are worth having. CS_LORA=0 answers "how does the technique do without its training",
# CS_LORA=1 answers "how good does it get with it".
if os.environ.get("CS_LORA", "1") == "1":
    model_manager.load_lora("models/t2v_14b_ntk20.ckpt", lora_alpha=1.0)
    print("[cinescale] LoRA loaded (trained configuration)")
else:
    print("[cinescale] NO LoRA -- training-free configuration")

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)
from xfuser.core.distributed import (initialize_model_parallel,
                                     init_distributed_environment)
init_distributed_environment(
    rank=dist.get_rank(), world_size=dist.get_world_size())

initialize_model_parallel(
    sequence_parallel_degree=dist.get_world_size(),
    ring_degree=1,
    ulysses_degree=dist.get_world_size(),
)
torch.cuda.set_device(dist.get_rank())

pipe = WanVideoPipeline_Pro.from_model_manager(model_manager, 
                                           torch_dtype=torch.bfloat16, 
                                           device=f"cuda:{dist.get_rank()}", 
                                           use_usp=True if dist.get_world_size() > 1 else False)
pipe.enable_vram_management(num_persistent_param_in_dit=None) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

# Overridable so one script covers 2K and 4K. CineScale generates at `height/width`
# and upscales to `target_*`, so the base stays at the model's native scale like ours.
height = int(os.environ.get("CS_HEIGHT", 720))
width = int(os.environ.get("CS_WIDTH", 1280))
target_height = int(os.environ.get("CS_TARGET_H", 1440))
target_width = int(os.environ.get("CS_TARGET_W", 2560))

# Text-to-video
videos = pipe(
    prompt=os.environ.get("CS_PROMPT", "a lion is roaring in the wild"),
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    num_inference_steps=50,
    seed=0, tiled=True,
    height=height, 
    width=width,
    num_frames=int(os.environ.get("CS_FRAMES", 41)),
    cfg_scale=5.0,
    sigma_shift=7.0,
    new_shift=11.0,
    ntk_first=20.0,
    ntk_second=25.0,
    target_height=target_height,
    target_width=target_width,
)
if dist.get_rank() == 0:
    for video_id, video in enumerate(videos):
        save_video(video, os.environ.get("CS_OUT", "cs_t2v_out.mp4"), fps=8)
