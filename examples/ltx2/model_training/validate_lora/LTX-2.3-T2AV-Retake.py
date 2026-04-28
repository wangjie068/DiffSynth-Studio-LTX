import ast
import glob
import os

import torch

from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.audio import read_audio
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2


def get_env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def get_env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def get_env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def find_lora_path():
    lora_path = os.environ.get("LORA_PATH")
    if lora_path:
        return lora_path

    lora_dir = os.environ.get("LORA_DIR", "models/train/LTX2.3-T2AV-Retake_lora")
    candidates = glob.glob(os.path.join(lora_dir, "epoch-*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No LoRA checkpoint found in {lora_dir}. Set LORA_PATH=/path/to/epoch-*.safetensors."
        )

    def epoch_id(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return int(name.split("-")[-1])

    return max(candidates, key=epoch_id)


def parse_regions(value):
    if value is None or value.strip() == "":
        return None
    regions = ast.literal_eval(value)
    return [(float(start), float(end)) for start, end in regions]


def load_input_video(path, height, width, num_frames):
    video = VideoData(path, height=height, width=width).raw_data()[:num_frames]
    if len(video) != num_frames:
        raise ValueError(f"Input video has {len(video)} frames, but expected {num_frames}.")
    return video


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

use_two_stage = get_env_bool("USE_TWO_STAGE", False)
model_configs = [
    ModelConfig(
        model_id=os.environ.get("TEXT_ENCODER_MODEL_ID", "google/gemma-3-12b-it-qat-q4_0-unquantized"),
        origin_file_pattern="model-*.safetensors",
        **vram_config,
    ),
    ModelConfig(
        model_id=os.environ.get("LTX_MODEL_ID", "DiffSynth-Studio/LTX-2.3-Repackage"),
        origin_file_pattern="transformer.safetensors",
        **vram_config,
    ),
    ModelConfig(
        model_id=os.environ.get("LTX_MODEL_ID", "DiffSynth-Studio/LTX-2.3-Repackage"),
        origin_file_pattern="text_encoder_post_modules.safetensors",
        **vram_config,
    ),
    ModelConfig(
        model_id=os.environ.get("LTX_MODEL_ID", "DiffSynth-Studio/LTX-2.3-Repackage"),
        origin_file_pattern="video_vae_encoder.safetensors",
        **vram_config,
    ),
    ModelConfig(
        model_id=os.environ.get("LTX_MODEL_ID", "DiffSynth-Studio/LTX-2.3-Repackage"),
        origin_file_pattern="video_vae_decoder.safetensors",
        **vram_config,
    ),
    ModelConfig(
        model_id=os.environ.get("LTX_MODEL_ID", "DiffSynth-Studio/LTX-2.3-Repackage"),
        origin_file_pattern="audio_vae_encoder.safetensors",
        **vram_config,
    ),
    ModelConfig(
        model_id=os.environ.get("LTX_MODEL_ID", "DiffSynth-Studio/LTX-2.3-Repackage"),
        origin_file_pattern="audio_vae_decoder.safetensors",
        **vram_config,
    ),
    ModelConfig(
        model_id=os.environ.get("LTX_MODEL_ID", "DiffSynth-Studio/LTX-2.3-Repackage"),
        origin_file_pattern="audio_vocoder.safetensors",
        **vram_config,
    ),
]
if use_two_stage:
    model_configs.append(
        ModelConfig(
            model_id=os.environ.get("UPSAMPLER_MODEL_ID", "Lightricks/LTX-2.3"),
            origin_file_pattern=os.environ.get("UPSAMPLER_PATTERN", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            **vram_config,
        )
    )

stage2_lora_config = None
if use_two_stage:
    stage2_lora_config = ModelConfig(
        model_id=os.environ.get("STAGE2_LORA_MODEL_ID", "Lightricks/LTX-2.3"),
        origin_file_pattern=os.environ.get("STAGE2_LORA_PATTERN", "ltx-2.3-22b-distilled-lora-384.safetensors"),
    )

pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs,
    tokenizer_config=ModelConfig(
        model_id=os.environ.get("TEXT_ENCODER_MODEL_ID", "google/gemma-3-12b-it-qat-q4_0-unquantized")
    ),
    stage2_lora_config=stage2_lora_config,
)

lora_path = find_lora_path()
print(f"[Validate] Loading LoRA: {lora_path}")
pipe.load_lora(pipe.dit, lora_path)

prompt = os.environ.get(
    "PROMPT",
    "Generate natural background music and sound effects that match the input video without changing the video content.",
)
negative_prompt = os.environ.get(
    "NEGATIVE_PROMPT",
    "silent or muted audio, distorted audio, robotic voice, echo, background noise, off-sync audio, "
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, changed video content, "
    "blurry, flickering, artifacts",
)

height = get_env_int("HEIGHT", 384)
width = get_env_int("WIDTH", 672)
num_frames = get_env_int("NUM_FRAMES", 81)
frame_rate = get_env_float("FRAME_RATE", 24)
seed = get_env_int("SEED", 43)
num_inference_steps = get_env_int("NUM_INFERENCE_STEPS", 30)
cfg_scale = get_env_float("CFG_SCALE", 3.0)
duration = num_frames / frame_rate

input_video_path = os.environ.get("INPUT_VIDEO", "data/example_video_dataset/ltx2/video.mp4")
retake_video_regions = parse_regions(os.environ.get("RETAKE_VIDEO_REGIONS"))
retake_audio_regions = parse_regions(os.environ.get("RETAKE_AUDIO_REGIONS"))

retake_video = load_input_video(input_video_path, height, width, num_frames)
retake_audio = None
audio_sample_rate = None
retake_audio_path = os.environ.get("RETAKE_AUDIO")
if retake_audio_path:
    retake_audio, audio_sample_rate = read_audio(
        retake_audio_path,
        start_time=get_env_float("RETAKE_AUDIO_START", 0),
        duration=get_env_float("RETAKE_AUDIO_DURATION", duration),
    )

video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    retake_video=retake_video,
    retake_video_regions=retake_video_regions,
    retake_audio=retake_audio,
    audio_sample_rate=audio_sample_rate or 48000,
    retake_audio_regions=retake_audio_regions,
    seed=seed,
    height=height,
    width=width,
    num_frames=num_frames,
    frame_rate=frame_rate,
    cfg_scale=cfg_scale,
    num_inference_steps=num_inference_steps,
    tiled=True,
    use_two_stage_pipeline=use_two_stage,
)

output_path = os.environ.get("OUTPUT_PATH", "ltx2.3_t2av_retake_lora_validate.mp4")
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path=output_path,
    fps=frame_rate,
    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
)
print(f"[Validate] Saved: {output_path}")
