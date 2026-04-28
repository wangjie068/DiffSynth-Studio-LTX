import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
from diffsynth.utils.data.audio import read_audio
from diffsynth.utils.data import VideoData

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
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="transformer.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_encoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vae_encoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
)
pipe.load_lora(pipe.dit, "models/train/LTX2.3-T2AV-Retake_lora/epoch-0.safetensors")

prompt = "Generate natural background music and sound effects that match the input video without changing the video content."
negative_prompt = (
    "silent or muted audio, distorted audio, robotic voice, echo, background noise, off-sync audio, "
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, changed video content, "
    "blurry, flickering, artifacts"
)

height, width, num_frames, frame_rate = 384, 672, 81, 24
input_video_path = "data/example_video_dataset/ltx2/video.mp4"
retake_video = VideoData(input_video_path, height=height, width=width).raw_data()[:num_frames]

# Optional audio retake. Set this to a real audio/video path and uncomment retake_audio_regions for local audio editing.
retake_audio_path = None
retake_audio, audio_sample_rate = None, 48000
if retake_audio_path is not None:
    retake_audio, audio_sample_rate = read_audio(retake_audio_path, duration=num_frames / frame_rate)

video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    retake_video=retake_video,
    # retake_video_regions=[(0, num_frames / frame_rate)],
    retake_audio=retake_audio,
    audio_sample_rate=audio_sample_rate,
    # retake_audio_regions=[(3, 4)],
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    frame_rate=frame_rate,
    tiled=True,
    cfg_scale=3.0,
    num_inference_steps=30,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path="ltx2.3_t2av_retake_lora.mp4",
    fps=frame_rate,
    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
)
