# Local LLM Deploy

本地大语言模型部署工具，支持 vLLM 推理服务、HuggingFace 模型检索与国内加速下载。

## 功能

- **vLLM 推理服务** - OpenAI API 兼容的本地推理服务
- **模型检索** - 从 HuggingFace Hub 搜索模型（按任务类型、热门度等）
- **国内加速下载** - 支持 HF-Mirror 和 ModelScope 镜像源
- **TUI 界面** - 交互式终端界面，方便搜索和下载模型

## 快速开始

### 1. 环境配置

```bash
./scripts/setup_env.sh
source venv/bin/activate
```

### 2. 配置国内镜像（可选）

```bash
./scripts/config_mirror.sh --persist
```

### 3. 搜索模型

```bash
# 命令行方式
python scripts/hf_search.py --task embedding --limit 10
python scripts/hf_search.py --search "qwen" --trending

# TUI 界面
python scripts/tui.py
```

### 4. 下载模型

```bash
python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct
python scripts/download_model.py BAAI/bge-large-zh-v1.5 --source modelscope
```

### 5. 启动推理服务

```bash
./scripts/start_vllm.sh Qwen/Qwen2.5-7B-Instruct 2 8000
```

### 6. 测试 API

```bash
python scripts/test_api.py "你好，请介绍一下你自己"
```

## 脚本说明

| 脚本 | 说明 |
|------|------|
| `setup_env.sh` | 初始化环境，安装依赖 |
| `config_mirror.sh` | 配置国内镜像源 |
| `hf_search.py` | 命令行模型检索 |
| `download_model.py` | 模型下载（支持国内源） |
| `tui.py` | 交互式 TUI 界面 |
| `start_vllm.sh` | 启动 vLLM 推理服务 |
| `test_api.py` | 测试 OpenAI 兼容 API |

## 任务类型别名

| 别名 | HuggingFace Task |
|------|------------------|
| embedding | feature-extraction |
| llm, chat | text-generation |
| image | image-classification |
| asr, speech | automatic-speech-recognition |
| tts | text-to-speech |

## 国内下载源

| 源 | 地址 | 说明 |
|----|------|------|
| HF-Mirror | hf-mirror.com | HuggingFace 镜像，推荐 |
| ModelScope | modelscope.cn | 阿里云魔搭社区 |

## 依赖

- Python 3.10+
- CUDA 11.8+ (GPU 推理)
- vLLM
- huggingface_hub
- modelscope
- textual

## License

MIT
