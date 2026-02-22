#!/usr/bin/env python3
"""Standalone inference script for local Qwen3-VL checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MIN_VRAM_BYTES = 18 * 1024**3  # Practical lower bound for 8B VL bf16 inference.


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if not torch.cuda.is_available():
        return "cpu"

    candidates: list[tuple[str, int]] = []
    for idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        candidates.append((f"cuda:{idx}", free_bytes))

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_device, best_free = candidates[0]

    if best_free < MIN_VRAM_BYTES:
        print(
            "[WARN] 没有 GPU 的空闲显存达到 18GB，可能触发 OOM。将使用空闲最多的设备："
            f"{best_device} ({best_free / 1024**3:.2f} GiB free)"
        )

    return best_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL 本地推理脚本")
    parser.add_argument("--model-dir", default=".", help="模型目录（含 config.json/index/shards）")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--prompt", required=True, help="用户文本提示词")
    parser.add_argument("--device", default="auto", help="推理设备，例如 auto/cuda:1/cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    image_path = Path(args.image)

    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    device = pick_device(args.device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"[INFO] device={device}, dtype={dtype}")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print("\n===== 模型回复 =====")
    print(response.strip())


if __name__ == "__main__":
    main()
