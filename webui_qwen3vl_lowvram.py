#!/usr/bin/env python3
"""Gradio Web UI for local Qwen3-VL checkpoints with low-VRAM quantization."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MIN_VRAM_BYTES = 6 * 1024**3


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
            "[WARN] 没有 GPU 的空闲显存达到 6GB，仍可能触发 OOM。将使用空闲最多的设备："
            f"{best_device} ({best_free / 1024**3:.2f} GiB free)"
        )

    return best_device


class Qwen3VLLowVramRunner:
    def __init__(
        self,
        model_dir: Path,
        device: str,
        precision: str,
        cpu_offload: bool,
    ) -> None:
        self.model_dir = model_dir
        self.device = pick_device(device)
        self.precision = precision
        self.cpu_offload = cpu_offload

        print(f"[INFO] Loading model from {self.model_dir}")
        print(
            f"[INFO] device={self.device}, precision={self.precision}, "
            f"cpu_offload={self.cpu_offload}"
        )

        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True)

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        quant_config = self._build_quantization_config()
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto" if self.device.startswith("cuda") else "cpu"
        else:
            model_kwargs["torch_dtype"] = self._resolve_dtype()

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            **model_kwargs,
        )

        if quant_config is None:
            self.model = self.model.to(self.device)

        self.model.eval()

    def _resolve_dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "fp32":
            return torch.float32

        if self.device.startswith("cuda"):
            return torch.float16
        return torch.float32

    def _build_quantization_config(self) -> BitsAndBytesConfig | None:
        if self.precision == "auto":
            if self.device.startswith("cuda"):
                self.precision = "int4"
            else:
                return None

        if self.precision not in {"int8", "int4"}:
            return None

        if not self.device.startswith("cuda"):
            raise ValueError("int8/int4 量化需要 CUDA GPU。请改用 fp32 或 fp16。")

        if self.precision == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=self.cpu_offload,
            )

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=self.cpu_offload,
        )

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
        target_device = self.device if self.device.startswith("cuda") else "cpu"
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

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
    parser = argparse.ArgumentParser(description="博视医疗Robotrak EyeGPT（低显存版）")
    parser.add_argument("--model-dir", default=".", help="模型目录（含 config.json/index/shards）")
    parser.add_argument("--device", default="auto", help="推理设备，例如 auto/cuda:0/cpu")
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32", "int8", "int4"],
        help="推理精度：6GB 显存建议 auto/int4",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="启用 CPU offload（显存不足时更稳，但速度更慢）",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Gradio 监听地址（默认仅本机回环，避免意外暴露到公网）",
    )
    parser.add_argument("--port", type=int, default=7860, help="Gradio 监听端口")
    parser.add_argument("--share", action="store_true", help="启用 Gradio share 链接")
    return parser.parse_args()


def build_ui(runner: Qwen3VLLowVramRunner) -> gr.Blocks:
    with gr.Blocks(title="博视医疗Robotrak EyeGPT") as demo:
        gr.Markdown("# 博视医疗Robotrak EyeGPT")
        gr.Markdown("上传图片并输入提示词，点击 **生成** 后查看模型回复。")

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

    runner = Qwen3VLLowVramRunner(
        model_dir=model_dir,
        device=args.device,
        precision=args.precision,
        cpu_offload=args.cpu_offload,
    )
    demo = build_ui(runner)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
