import argparse
import csv
import json
import math
import os
import sys


def is_missing_value(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "null"):
        return True
    return False


def resolve_path(base_path, value):
    if is_missing_value(value):
        return None
    value = str(value)
    return value if os.path.isabs(value) else os.path.join(base_path, value)


def load_metadata(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_regions(value):
    if is_missing_value(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("retake_audio_regions must be a list or JSON list string.")
    regions = []
    for region in value:
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            raise ValueError(f"Invalid region: {region}")
        start, end = float(region[0]), float(region[1])
        regions.append((start, end))
    return regions


def get_video_duration(path, frame_rate):
    import imageio

    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        raw_fps = float(metadata.get("fps") or frame_rate)
        total_frames = int(reader.count_frames())
        duration = float(metadata.get("duration") or total_frames / raw_fps)
        return total_frames, raw_fps, duration
    finally:
        reader.close()


def get_training_duration(video_duration, num_frames, frame_rate):
    total_available_frames = int(math.floor(video_duration * frame_rate))
    training_frames = num_frames
    if total_available_frames < training_frames:
        training_frames = total_available_frames
        while training_frames > 1 and training_frames % 8 != 1:
            training_frames -= 1
    if training_frames <= 1:
        raise ValueError("video is too short to provide a valid 8n+1 training clip")
    return training_frames / frame_rate


def check_audio(path):
    import torchaudio

    waveform, sample_rate = torchaudio.load(path)
    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise ValueError("audio has zero samples")
    return waveform.shape[-1] / sample_rate


def validate_regions(regions, training_duration, min_region_seconds):
    if len(regions) == 0:
        raise ValueError("retake_audio is provided, but retake_audio_regions is missing or empty.")
    for start, end in regions:
        if start < 0:
            raise ValueError(f"region starts before 0: {(start, end)}")
        if end <= start:
            raise ValueError(f"region end must be greater than start: {(start, end)}")
        if end > training_duration:
            raise ValueError(f"region exceeds training clip duration {training_duration:.3f}s: {(start, end)}")
        if end - start < min_region_seconds:
            raise ValueError(
                f"region is too short for stable latent coverage, minimum is {min_region_seconds:.3f}s: {(start, end)}"
            )


def build_parser():
    parser = argparse.ArgumentParser(description="Precheck LTX2 audio generation/edit training metadata.")
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--frame_rate", type=float, default=24)
    parser.add_argument("--min_edit_region_seconds", type=float, default=0.25)
    parser.add_argument("--max_errors", type=int, default=20)
    return parser


def main():
    args = build_parser().parse_args()
    if not os.path.exists(args.dataset_metadata_path):
        raise FileNotFoundError(f"metadata not found: {args.dataset_metadata_path}")

    rows = load_metadata(args.dataset_metadata_path)
    errors = []
    generation_count = 0
    edit_count = 0

    for index, row in enumerate(rows):
        try:
            video_path = resolve_path(args.dataset_base_path, row.get("video"))
            input_audio_path = resolve_path(args.dataset_base_path, row.get("input_audio"))
            retake_audio_path = resolve_path(args.dataset_base_path, row.get("retake_audio"))

            if video_path is None:
                raise ValueError("missing video")
            if input_audio_path is None:
                raise ValueError("missing input_audio")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"video not found: {video_path}")
            if not os.path.exists(input_audio_path):
                raise FileNotFoundError(f"input_audio not found: {input_audio_path}")

            _, _, video_duration = get_video_duration(video_path, args.frame_rate)
            training_duration = get_training_duration(video_duration, args.num_frames, args.frame_rate)
            input_audio_duration = check_audio(input_audio_path)
            if input_audio_duration <= 0:
                raise ValueError("input_audio duration is zero")

            if retake_audio_path is None:
                generation_count += 1
            else:
                if not os.path.exists(retake_audio_path):
                    raise FileNotFoundError(f"retake_audio not found: {retake_audio_path}")
                check_audio(retake_audio_path)
                regions = parse_regions(row.get("retake_audio_regions"))
                validate_regions(regions, training_duration, args.min_edit_region_seconds)
                edit_count += 1
        except Exception as error:
            errors.append(f"row {index}: {error}")
            if len(errors) >= args.max_errors:
                break

    print(f"[Precheck] Samples: {len(rows)}")
    print(f"[Precheck] Generation samples: {generation_count}")
    print(f"[Precheck] Edit samples: {edit_count}")
    if errors:
        print("[Precheck] Errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    print("[Precheck] OK")


if __name__ == "__main__":
    main()
