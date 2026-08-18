#!/usr/bin/env python3
"""Test SARM-style subtask annotation using Qwen2-VL-7B on Intel XPU.

Usage:
    conda run -n vllm-xpu python test_sarm_annotation.py
    conda run -n vllm-xpu python test_sarm_annotation.py --episode 5
    conda run -n vllm-xpu python test_sarm_annotation.py --video /path/to/video.mp4
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# Qwen2.5-Omni-7B is already cached — no download needed.
# Fallback: Qwen2-VL-7B-Instruct (needs ~15GB download)
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

# Dataset path (host path via sudo-accessible docker volume, or inside container)
DATASET_PATH = Path(
    "/var/lib/docker/volumes/docker_physical-ai-studio-storage/_data/datasets"
    "/bcdf34ee-97b1-4c89-8be7-c3045b24ae49"
)
VIDEO_SUBDIR = "videos/observation.images.topview/chunk-000"

SUBTASK_PROMPT = """You are analyzing a robot arm sorting task. The robot picks up colored blocks
and places them in the correct bin. There are two types of blocks:
- Blocks with a RED X mark → go in the left bin (recycle bin)
- Blocks with a BLUE X mark → go in the right bin (keep bin)

Watch these video frames (sampled at ~1 FPS) and identify the subtasks with timestamps.

Expected subtask sequence:
1. Approach block - robot arm moves toward the block
2. Grasp block - robot gripper closes on the block
3. Lift block - robot lifts the block up
4. Move to bin - robot carries block toward the target bin
5. Release block - robot opens gripper, drops block in bin
6. Return home - robot returns to neutral position

For each frame (listed as frame index), identify which subtask is happening.
Output as JSON:
{
  "subtasks": [
    {"frame": <int>, "subtask": "<name>", "description": "<what you see>"},
    ...
  ],
  "block_color": "<red|blue|unknown>",
  "task_success": <true|false|null>
}
Only output the JSON, nothing else."""


def extract_frames(video_path: Path, fps: float = 1.0, max_frames: int = 30) -> list[Path]:
    """Extract frames from video at given FPS using ffmpeg."""
    tmpdir = Path(tempfile.mkdtemp())
    frame_pattern = tmpdir / "frame_%04d.jpg"
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(frame_pattern),
        "-y", "-loglevel", "error"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    frames = sorted(tmpdir.glob("frame_*.jpg"))
    print(f"  Extracted {len(frames)} frames from {video_path.name}")
    return frames


def frames_to_messages(frames: list[Path], prompt: str) -> list[dict]:
    """Build Qwen2-VL chat message with inline base64 images."""
    content = []
    for i, frame_path in enumerate(frames):
        with open(frame_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "text",
            "text": f"Frame {i}:"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_model():
    """Load Qwen2.5-Omni-7B (or Qwen2-VL-7B) on XPU."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    device = "xpu" if torch.xpu.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "xpu" else torch.float32

    print(f"Loading {MODEL_ID} on {device} ({dtype})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Qwen2.5-Omni has a thinker + talker architecture; load vision-only thinker
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            enable_audio_output=False,  # text-only output
        )
    except (ImportError, TypeError):
        # Fallback to AutoModel for older transformers
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    model.eval()
    if torch.xpu.is_available():
        print(f"  Model loaded. Memory allocated: {torch.xpu.memory_allocated(0)/1e9:.2f} GB")
    return model, processor, device


def annotate_episode(model, processor, device, video_path: Path) -> dict:
    """Run annotation on a single episode video."""
    print(f"\nAnnotating: {video_path.name}")
    frames = extract_frames(video_path, fps=1.0, max_frames=30)
    if not frames:
        return {"error": "no frames extracted"}

    messages = frames_to_messages(frames, SUBTASK_PROMPT)

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process with vision
    # Qwen2VL processor accepts image inputs separately from text
    from PIL import Image
    images = [Image.open(f) for f in frames]

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"  Running inference on {len(frames)} frames...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True)

    print(f"  Response:\n{response}\n")

    # Try to parse JSON
    try:
        # Find JSON block
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
        else:
            result = {"raw_response": response}
    except json.JSONDecodeError:
        result = {"raw_response": response}

    # Cleanup temp frames
    for f in frames:
        f.unlink()
    frames[0].parent.rmdir()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0, help="Episode index (0-based)")
    parser.add_argument("--video", type=str, default=None, help="Direct path to mp4 file")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset UUID (default: bcdf34ee)")
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        dataset_path = Path(args.dataset) if args.dataset else DATASET_PATH
        video_dir = dataset_path / VIDEO_SUBDIR
        videos = sorted(video_dir.glob("file-*.mp4"))
        if not videos:
            # Try with sudo-accessible path
            print(f"No videos found in {video_dir}")
            print("Try running: sudo python test_sarm_annotation.py")
            sys.exit(1)
        if args.episode >= len(videos):
            print(f"Episode {args.episode} not found. Available: 0-{len(videos)-1}")
            sys.exit(1)
        video_path = videos[args.episode]

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    model, processor, device = load_model()
    result = annotate_episode(model, processor, device, video_path)

    print("\n=== ANNOTATION RESULT ===")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    main()
