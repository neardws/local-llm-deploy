# Local LLM Deploy

本地大语言模型部署工具，支持 vLLM 推理服务、HuggingFace 模型检索与国内加速下载。

## 功能特性

- **vLLM 推理服务** - OpenAI API 兼容的本地推理服务
- **模型检索** - 从 HuggingFace Hub 搜索模型（按任务类型、热门度等）
- **国内加速下载** - 支持 HF-Mirror 和 ModelScope 镜像源
- **TUI 交互界面** - 美观的终端界面，支持中英文切换

## TUI 界面预览

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Search Models  │  Quick Browse   │    Download     │
│                 │ [🔥Hot][💬LLM]  │                 │
│ Keyword: ____   │ [🔢Emb][⭐Pick] │ Model ID: ___   │
│ Task: [All ▼]   │   [EN] [中]     │ Source: [HF ▼]  │
│ Sort: [Down ▼]  │                 │ [Download]      │
└─────────────────┴─────────────────┴─────────────────┘
┌─────────────────────────────────────────────────────┐
│ # │ Model ID          │Params│VRAM │Local│DL │Desc │
│ 1 │ Qwen/Qwen2.5-7B   │ 7.6B │18GB │ Yes │5M │Chat │
│ 2 │ meta-llama/Llama  │ 8.0B │19GB │ Yes │3M │Chat │
└─────────────────────────────────────────────────────┘
```

### TUI 功能

| 按钮 | 功能 |
|------|------|
| 🔥 Hot | 实时热门模型 (Trending) |
| 💬 LLM | 热门文本生成/对话模型 |
| 🔢 Embed | 热门向量嵌入模型 |
| ⭐ Picks | AI 精选推荐模型 |
| EN / 中 | 中英文界面切换 |

### 模型信息列

| 列 | 说明 |
|----|------|
| Params | 模型参数量（从 HF API 获取精确值） |
| VRAM | 估算显存需求 |
| Local | 是否可本地部署（基于 GPU 显存判断） |
| Downloads | 下载量 |
| Description | 模型描述（自动生成） |

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

### 3. 使用 TUI 界面（推荐）

```bash
python scripts/tui.py
```

### 4. 命令行搜索模型

```bash
# 搜索 embedding 模型
python scripts/hf_search.py --task embedding --limit 10

# 搜索关键词
python scripts/hf_search.py --search "qwen" --trending

# 搜索 LLM 模型
python scripts/hf_search.py --task llm --sort likes
```

### 5. 下载模型

```bash
# 使用 HF-Mirror 下载（默认，推荐）
python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct

# 使用 ModelScope 下载
python scripts/download_model.py BAAI/bge-large-zh-v1.5 --source modelscope

# 指定下载目录
python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct --dir ./models
```

### 6. 启动推理服务

```bash
# 启动 vLLM 服务 (模型名, 张量并行数, 端口)
./scripts/start_vllm.sh Qwen/Qwen2.5-7B-Instruct 2 8000
```

### 7. 测试 API

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
| embedding, embed | feature-extraction |
| llm, chat | text-generation |
| image | image-classification |
| asr, speech | automatic-speech-recognition |
| tts | text-to-speech |

## 国内下载源

| 源 | 地址 | 说明 |
|----|------|------|
| HF-Mirror | hf-mirror.com | HuggingFace 镜像，推荐 |
| ModelScope | modelscope.cn | 阿里云魔搭社区 |

## AI 精选模型 (Picks)

| 模型 | 说明 |
|------|------|
| deepseek-ai/DeepSeek-R1 | 顶级推理模型，媲美 o1 |
| Qwen/Qwen2.5-72B-Instruct | 最强开源通用大模型 |
| meta-llama/Llama-3.3-70B-Instruct | Meta 旗舰，多语言出色 |
| BAAI/bge-m3 | 最强多语言向量模型 |
| black-forest-labs/FLUX.1-dev | 最强文生图模型 |
| openai/whisper-large-v3 | 最强语音识别模型 |
| microsoft/phi-4 | 14B 小模型，性能超群 |

## 依赖

- Python 3.10+
- CUDA 11.8+ (GPU 推理)
- vLLM
- huggingface_hub
- modelscope
- textual

## License

MIT
