#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

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
MODEL_SOURCE_BASE_PATH="${MODEL_SOURCE_BASE_PATH:-/mnt/bn/genai-nebula/wangjie/Video_Gen_Long/DiffSynth-Studio/models}"
SHM_MODEL_BASE_PATH="${SHM_MODEL_BASE_PATH:-/dev/shm/diffsynth_model_links}"
MODEL_CACHE_MODE="${MODEL_CACHE_MODE:-symlink}"
RUN_DATA_PRECHECK="${RUN_DATA_PRECHECK:-1}"
MIN_EDIT_REGION_SECONDS="${MIN_EDIT_REGION_SECONDS:-0.25}"
HEIGHT="${HEIGHT:-}"
WIDTH="${WIDTH:-}"
MAX_PIXELS="${MAX_PIXELS:-921600}"
NUM_FRAMES="${NUM_FRAMES:-481}"
FRAME_RATE="${FRAME_RATE:-24}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-16}"
DATASET_REPEAT="${DATASET_REPEAT:-50}"
OUTPUT_NAME="${OUTPUT_NAME:-LTX2.3-T2AV-Retake_lora}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MODEL_RUNTIME_BASE_PATH="${SHM_MODEL_BASE_PATH}"
if [[ "${MODEL_CACHE_MODE}" == "none" ]]; then
  MODEL_RUNTIME_BASE_PATH="${MODEL_SOURCE_BASE_PATH}"
fi

VIDEO_SIZE_ARGS=(--max_pixels "${MAX_PIXELS}" --num_frames "${NUM_FRAMES}")
ACCELERATE_ARGS=(--num_processes "${NUM_PROCESSES}")
if [[ -n "${HEIGHT}" || -n "${WIDTH}" ]]; then
  if [[ -z "${HEIGHT}" || -z "${WIDTH}" ]]; then
    echo "HEIGHT and WIDTH must be set together. Leave both empty to enable dynamic resolution." >&2
    exit 1
  fi
  VIDEO_SIZE_ARGS+=(--height "${HEIGHT}" --width "${WIDTH}")
fi

log_step() {
  echo "[LTX2 Retake] $*" >&2
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
}

prepare_model_cache() {
  local source_dir="$1"
  local relative_dir="$2"
  local target_dir="${SHM_MODEL_BASE_PATH}/${relative_dir}"
  local ready_file="${target_dir}/.diffsynth_shm_ready"

  require_path "${source_dir}"
  if [[ -f "${ready_file}" ]]; then
    log_step "Model cache ready: ${target_dir}"
    return
  fi

  mkdir -p "$(dirname "${target_dir}")"
  if [[ "${MODEL_CACHE_MODE}" == "copy" ]]; then
    if [[ -L "${target_dir}" ]]; then
      echo "Cannot copy model cache over symlink: ${target_dir}" >&2
      echo "Use a fresh SHM_MODEL_BASE_PATH or set MODEL_CACHE_MODE=symlink." >&2
      exit 1
    fi
    log_step "Copying model cache to SHM: ${source_dir} -> ${target_dir}"
    mkdir -p "${target_dir}"
    cp -a "${source_dir}/." "${target_dir}/"
    touch "${ready_file}"
    log_step "Model cache copied: ${target_dir}"
  elif [[ "${MODEL_CACHE_MODE}" == "symlink" ]]; then
    if [[ ! -e "${target_dir}" && ! -L "${target_dir}" ]]; then
      log_step "Creating model cache symlink: ${target_dir} -> ${source_dir}"
      ln -s "${source_dir}" "${target_dir}"
    elif [[ -L "${target_dir}" || -f "${ready_file}" ]]; then
      log_step "Model cache symlink/path exists: ${target_dir}"
    else
      echo "Existing non-symlink cache path may be incomplete: ${target_dir}" >&2
      echo "Use a fresh SHM_MODEL_BASE_PATH, remove the old cache manually, or set MODEL_CACHE_MODE=none." >&2
      exit 1
    fi
  elif [[ "${MODEL_CACHE_MODE}" == "none" ]]; then
    log_step "Skipping model cache for ${source_dir}"
    return
  else
    echo "Unsupported MODEL_CACHE_MODE=${MODEL_CACHE_MODE}. Use copy, symlink, or none." >&2
    exit 1
  fi
}

log_step "Config: max_pixels=${MAX_PIXELS}, num_frames=${NUM_FRAMES}, frame_rate=${FRAME_RATE}, precheck=${RUN_DATA_PRECHECK}, cache_mode=${MODEL_CACHE_MODE}, num_processes=${NUM_PROCESSES}"

if [[ "${RUN_DATA_PRECHECK}" == "1" ]]; then
  log_step "Running dataset precheck: ${DATASET_METADATA_PATH}"
  python3 examples/ltx2/model_training/lora/precheck_audio_hybrid_dataset.py \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --num_frames "${NUM_FRAMES}" \
    --frame_rate "${FRAME_RATE}" \
    --min_edit_region_seconds "${MIN_EDIT_REGION_SECONDS}"
  log_step "Dataset precheck finished"
fi

log_step "Preparing model cache"
prepare_model_cache "${MODEL_SOURCE_BASE_PATH}/DiffSynth-Studio/LTX-2.3-Repackage" "DiffSynth-Studio/LTX-2.3-Repackage"
prepare_model_cache "${MODEL_SOURCE_BASE_PATH}/google/gemma-3-12b-it-qat-q4_0-unquantized" "google/gemma-3-12b-it-qat-q4_0-unquantized"

require_path "${MODEL_RUNTIME_BASE_PATH}/DiffSynth-Studio/LTX-2.3-Repackage/text_encoder_post_modules.safetensors"
require_path "${MODEL_RUNTIME_BASE_PATH}/DiffSynth-Studio/LTX-2.3-Repackage/video_vae_encoder.safetensors"
require_path "${MODEL_RUNTIME_BASE_PATH}/DiffSynth-Studio/LTX-2.3-Repackage/audio_vae_encoder.safetensors"
require_path "${MODEL_RUNTIME_BASE_PATH}/DiffSynth-Studio/LTX-2.3-Repackage/transformer.safetensors"
require_path "${MODEL_RUNTIME_BASE_PATH}/google/gemma-3-12b-it-qat-q4_0-unquantized"

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_RUNTIME_BASE_PATH}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

log_step "Starting data-process stage"
accelerate launch "${ACCELERATE_ARGS[@]}" examples/ltx2/model_training/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video,input_audio,retake_audio" \
  --extra_inputs "input_audio,retake_audio,retake_audio_regions" \
  "${VIDEO_SIZE_ARGS[@]}" \
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

log_step "Starting LoRA train stage"
accelerate launch "${ACCELERATE_ARGS[@]}" examples/ltx2/model_training/train.py \
  --dataset_base_path "./models/train/${OUTPUT_NAME}-splited-cache" \
  --data_file_keys "video,input_audio,retake_audio" \
  --extra_inputs "input_audio,retake_audio,retake_audio_regions" \
  "${VIDEO_SIZE_ARGS[@]}" \
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
