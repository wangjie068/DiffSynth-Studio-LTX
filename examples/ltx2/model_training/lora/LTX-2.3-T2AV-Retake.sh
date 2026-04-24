#!/usr/bin/env bash
set -euo pipefail

# Recommended metadata format: JSON or JSONL.
# Required fields for every sample:
#   video, input_audio
# Optional fields for mixed generation + audio-edit training:
#   prompt, frame_rate, retake_audio, retake_audio_regions
#
# Generation sample:
#   retake_audio = null (or omit it)
#   retake_audio_regions = []
#
# Edit sample:
#   retake_audio = original audio before editing
#   retake_audio_regions = [[start_sec, end_sec], ...]
#   input_audio = final target audio after editing

DATASET_BASE_PATH="${DATASET_BASE_PATH:-/mnt/bn/genai-nebula/wangjie/Video_Gen_Long/DiffSynth-Studio/data/train_data}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata_audio_hybrid.json}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-672}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FRAME_RATE="${FRAME_RATE:-24}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
DATASET_REPEAT="${DATASET_REPEAT:-50}"
OUTPUT_NAME="${OUTPUT_NAME:-LTX2.3-V2AV-AudioHybrid_lora}"

accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video,input_audio,retake_audio" \
  --extra_inputs "input_audio,retake_audio,retake_audio_regions" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --frame_rate "${FRAME_RATE}" \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:text_encoder_post_modules.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:video_vae_encoder.safetensors,DiffSynth-Studio/LTX-2.3-Repackage:audio_vae_encoder.safetensors,google/gemma-3-12b-it-qat-q4_0-unquantized:model-*.safetensors" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --video_loss_weight 0.0 \
  --audio_loss_weight 1.0 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/${OUTPUT_NAME}-splited-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank "${LORA_RANK}" \
  --use_gradient_checkpointing \
  --task "sft:data_process"

accelerate launch examples/ltx2/model_training/train.py \
  --dataset_base_path "./models/train/${OUTPUT_NAME}-splited-cache" \
  --data_file_keys "video,input_audio,retake_audio" \
  --extra_inputs "input_audio,retake_audio,retake_audio_regions" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --frame_rate "${FRAME_RATE}" \
  --dataset_repeat "${DATASET_REPEAT}" \
  --model_id_with_origin_paths "DiffSynth-Studio/LTX-2.3-Repackage:transformer.safetensors" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --video_loss_weight 0.0 \
  --audio_loss_weight 1.0 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/${OUTPUT_NAME}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_k,to_q,to_v,to_out.0" \
  --lora_rank "${LORA_RANK}" \
  --use_gradient_checkpointing \
  --task "sft:train"
