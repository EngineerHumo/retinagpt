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

在远程服务器上执行（默认仅监听 `127.0.0.1`，更安全）：

```bash
python webui_qwen3vl.py --model-dir . --port 7860
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

> 安全说明：如果你显式设置 `--host 0.0.0.0`，服务会监听所有网卡地址；当服务器安全组/防火墙放通端口时，该 UI 可能被公网直接访问。除非你明确需要局域网/公网访问，否则建议保持默认 `127.0.0.1` 并配合 SSH 端口转发使用。

## 4. 常用参数

- `--device auto|cuda:0|cpu`
- `--port 7860`
- `--share`（需要公网 share 链接时启用）
