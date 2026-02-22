# RetinaGPT（Qwen3-VL 本地模型）

本仓库当前包含本地 Qwen3-VL 模型权重与推理脚本，支持：

- 命令行推理：`infer_qwen3vl.py`
- Web UI 推理：`webui_qwen3vl.py`（Gradio）

## 1. 安装依赖

```bash
pip install torch transformers pillow gradio
```

> 如需 GPU 推理，请先按你的 CUDA 版本安装对应的 PyTorch。

## 2. 命令行推理

```bash
python infer_qwen3vl.py \
  --model-dir . \
  --image ./demo.jpg \
  --prompt "请描述图像中的主要内容"
```

## 3. 启动 Web UI（远程服务器）

在远程服务器上执行：

```bash
python webui_qwen3vl.py --model-dir . --host 0.0.0.0 --port 7860
```

然后在你的本地电脑建立 SSH 端口转发：

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server_ip>
```

最后在本地浏览器打开：

```text
http://127.0.0.1:7860
```

你可以在页面中上传图片、输入文字并查看模型回复。

## 4. 常用参数

- `--device auto|cuda:0|cpu`
- `--port 7860`
- `--share`（需要公网 share 链接时启用）
