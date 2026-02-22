#!/usr/bin/env python3
"""Gradio Web UI for local Qwen3-VL checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
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


class Qwen3VLRunner:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = pick_device(device)
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        print(f"[INFO] Loading model from {self.model_dir}")
        print(f"[INFO] device={self.device}, dtype={self.dtype}")

        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        if image is None:
            return "错误：请先上传图片。"
        if not prompt.strip():
            return "错误：请输入文本提示词。"

        image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL Gradio Web UI")
    parser.add_argument("--model-dir", default=".", help="模型目录（含 config.json/index/shards）")
    parser.add_argument("--device", default="auto", help="推理设备，例如 auto/cuda:1/cpu")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio 监听地址")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 监听端口")
    parser.add_argument("--share", action="store_true", help="启用 Gradio share 链接")
    return parser.parse_args()


def build_ui(runner: Qwen3VLRunner) -> gr.Blocks:
    with gr.Blocks(title="Qwen3-VL Local WebUI") as demo:
        gr.Markdown("# Qwen3-VL 本地推理 WebUI")
        gr.Markdown(
            "上传图片并输入提示词，点击 **生成** 后查看模型回复。"
            "模型只会在服务启动时加载一次。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="输入图片")
                prompt_input = gr.Textbox(lines=5, label="文本提示词", placeholder="请描述这张图片...")

                max_new_tokens = gr.Slider(32, 1024, value=256, step=1, label="max_new_tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="top_p")
                top_k = gr.Slider(1, 100, value=20, step=1, label="top_k")

                run_btn = gr.Button("生成", variant="primary")

            with gr.Column(scale=1):
                output_text = gr.Textbox(lines=18, label="模型回复")

        run_btn.click(
            fn=runner.generate,
            inputs=[image_input, prompt_input, max_new_tokens, temperature, top_p, top_k],
            outputs=output_text,
        )

    return demo


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    runner = Qwen3VLRunner(model_dir=model_dir, device=args.device)
    demo = build_ui(runner)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
