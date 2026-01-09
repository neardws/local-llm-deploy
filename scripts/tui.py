#!/usr/bin/env python3
"""TUI for HuggingFace Model Search and Download"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Input,
    Button,
    DataTable,
    Static,
    Select,
    Label,
)
from textual.binding import Binding
from textual import work
from huggingface_hub import HfApi
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

TASK_OPTIONS = [
    ("All Tasks", ""),
    ("Embedding", "feature-extraction"),
    ("Text Generation (LLM)", "text-generation"),
    ("Image Classification", "image-classification"),
    ("ASR (Speech)", "automatic-speech-recognition"),
    ("Text-to-Speech", "text-to-speech"),
    ("Translation", "translation"),
    ("Summarization", "summarization"),
    ("Question Answering", "question-answering"),
]

SORT_OPTIONS = [
    ("Downloads", "downloads"),
    ("Likes", "likes"),
    ("Recently Created", "created"),
    ("Recently Modified", "modified"),
]

SOURCE_OPTIONS = [
    ("HF-Mirror (Recommended)", "hf-mirror"),
    ("ModelScope", "modelscope"),
]

QUANT_PATTERNS = ["gguf", "gptq", "awq", "bnb", "int4", "int8", "fp16", "bf16", "exl2"]
SIZE_PATTERNS = [
    (r"[-_](\d+)x(\d+\.?\d*)[bB]", lambda m: int(m.group(1)) * float(m.group(2)) * 1e9),  # 8x7B
    (r"[-_](\d+\.?\d*)[bB][-_]", 1e9),  # -7B- or _7B_
    (r"[-_](\d+\.?\d*)[bB]$", 1e9),     # ends with -7B
    (r"[-/](\d+\.?\d*)[bB][^a-zA-Z]", 1e9),  # /7B- or -7B-
    (r"(\d+\.?\d*)[bB]", 1e9),          # general 7B
    (r"[-_](\d+)[mM][-_]", 1e6),        # -350M-
    (r"[-_](\d+)[mM]$", 1e6),           # ends with -350M
]

LOCAL_VRAM_GB = 96  # 2x RTX 4090 D (48GB each)


def format_number(num: int) -> str:
    if num is None:
        return "N/A"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def format_size(size_bytes: int) -> str:
    if size_bytes is None:
        return "N/A"
    if size_bytes >= 1e12:
        return f"{size_bytes / 1e12:.1f}TB"
    elif size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f}GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f}MB"
    return f"{size_bytes / 1e3:.1f}KB"


def extract_params_from_name(model_id: str, tags: list) -> tuple[str, int]:
    """Extract parameter count from model name or tags, returns (display_str, raw_params)"""
    text = model_id + " " + " ".join(tags or [])
    for pattern, multiplier in SIZE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            if callable(multiplier):
                params = multiplier(match)
            else:
                params = float(match.group(1)) * multiplier
            if params >= 1e9:
                return f"{params / 1e9:.1f}B", int(params)
            elif params >= 1e6:
                return f"{params / 1e6:.0f}M", int(params)
    return "N/A", 0


def get_model_params(api: HfApi, model_id: str) -> int:
    """Get exact parameter count from model_info API"""
    try:
        info = api.model_info(model_id)
        if info.safetensors and info.safetensors.total:
            return info.safetensors.total
    except:
        pass
    return 0


def format_params(params: int) -> str:
    """Format parameter count for display"""
    if params <= 0:
        return "N/A"
    if params >= 1e9:
        return f"{params / 1e9:.1f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.0f}M"
    return f"{params / 1e3:.0f}K"


def extract_quant_info(model_id: str, tags: list) -> str:
    """Extract quantization info from model name or tags"""
    text = model_id.lower() + " " + " ".join(tags or []).lower()
    quants = []
    for q in QUANT_PATTERNS:
        if q in text:
            quants.append(q.upper())
    return ", ".join(quants) if quants else "No"


def estimate_vram(params: int, quant_info: str) -> str:
    """Estimate VRAM requirement based on params and quantization"""
    if params <= 0:
        return "N/A"
    
    if "INT4" in quant_info or "GPTQ" in quant_info or "AWQ" in quant_info or "GGUF" in quant_info:
        bytes_per_param = 0.5
    elif "INT8" in quant_info or "BNB" in quant_info:
        bytes_per_param = 1
    elif "FP16" in quant_info or "BF16" in quant_info:
        bytes_per_param = 2
    else:
        bytes_per_param = 2
    
    vram_bytes = params * bytes_per_param * 1.2
    vram_gb = vram_bytes / 1e9
    return f"{vram_gb:.1f}GB"


def can_run_locally(vram_str: str, local_vram: float = LOCAL_VRAM_GB) -> str:
    """Check if model can run on local GPU"""
    if vram_str == "N/A":
        return "?"
    try:
        vram = float(vram_str.replace("GB", ""))
        if vram <= local_vram:
            return "Yes"
        elif vram <= local_vram * 2:
            return "Maybe"
        else:
            return "No"
    except:
        return "?"


class SearchPanel(Static):
    """Search panel with filters"""

    def compose(self) -> ComposeResult:
        yield Label("Search Models", classes="panel-title")
        yield Input(placeholder="Search keyword (e.g., qwen, llama)", id="search-input")
        yield Horizontal(
            Vertical(
                Label("Task Type"),
                Select(TASK_OPTIONS, id="task-select", value=""),
            ),
            Vertical(
                Label("Sort By"),
                Select(SORT_OPTIONS, id="sort-select", value="downloads"),
            ),
            classes="filter-row",
        )
        yield Horizontal(
            Button("Search", id="search-btn", variant="primary"),
            Button("Trending", id="trending-btn", variant="default"),
            classes="button-row",
        )


class ResultsPanel(Static):
    """Results table panel"""

    def compose(self) -> ComposeResult:
        yield Label("Search Results", classes="panel-title")
        yield DataTable(id="results-table")


# Language settings
LANG = "en"  # "en" or "zh"

LABELS = {
    "en": {
        "quick_browse": "Quick Browse",
        "trending": "Trending",
        "week": "This Week",
        "month": "This Month",
        "llm_picks": "AI Picks",
        "search": "Search",
        "download": "Download",
        "task_type": "Task Type",
        "sort_by": "Sort By",
        "language": "Lang",
        "model_id": "Model ID",
        "searching": "Searching...",
        "loading": "Loading...",
        "found": "Found",
        "models": "models",
    },
    "zh": {
        "quick_browse": "快速浏览",
        "trending": "热门",
        "week": "本周新品",
        "month": "本月精选",
        "llm_picks": "AI 推荐",
        "search": "搜索",
        "download": "下载",
        "task_type": "任务类型",
        "sort_by": "排序方式",
        "language": "语言",
        "model_id": "模型ID",
        "searching": "搜索中...",
        "loading": "加载中...",
        "found": "找到",
        "models": "个模型",
    },
}

# LLM-curated interesting models with descriptions
LLM_PICKS = {
    "deepseek-ai/DeepSeek-R1": {
        "en": "Top reasoning model, rivals o1",
        "zh": "顶级推理模型，媲美o1",
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "en": "Best open-source LLM for general tasks",
        "zh": "最强开源通用大模型",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "en": "Meta's flagship, great multilingual",
        "zh": "Meta旗舰，多语言出色",
    },
    "BAAI/bge-m3": {
        "en": "Best multilingual embedding model",
        "zh": "最强多语言向量模型",
    },
    "black-forest-labs/FLUX.1-dev": {
        "en": "State-of-the-art image generation",
        "zh": "最强文生图模型",
    },
    "openai/whisper-large-v3": {
        "en": "Best speech recognition model",
        "zh": "最强语音识别模型",
    },
    "microsoft/phi-4": {
        "en": "Compact 14B, punches above weight",
        "zh": "14B小模型，性能超群",
    },
    "google/gemma-2-27b-it": {
        "en": "Google's efficient instruction model",
        "zh": "谷歌高效指令模型",
    },
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "en": "Best MoE architecture model",
        "zh": "最佳MoE架构模型",
    },
    "stabilityai/stable-diffusion-3.5-large": {
        "en": "Latest Stable Diffusion for images",
        "zh": "最新SD文生图模型",
    },
}

# Model description templates based on pipeline_tag
MODEL_DESC_TEMPLATES = {
    "en": {
        "text-generation": "LLM for text generation and chat",
        "feature-extraction": "Embedding model for semantic search",
        "text-to-image": "Image generation from text prompts",
        "automatic-speech-recognition": "Speech to text transcription",
        "text-to-speech": "Text to speech synthesis",
        "translation": "Language translation model",
        "summarization": "Text summarization model",
        "question-answering": "Q&A and reading comprehension",
        "image-classification": "Image classification model",
        "object-detection": "Object detection in images",
        "default": "AI model",
    },
    "zh": {
        "text-generation": "文本生成/对话大模型",
        "feature-extraction": "向量嵌入模型",
        "text-to-image": "文生图模型",
        "automatic-speech-recognition": "语音识别模型",
        "text-to-speech": "语音合成模型",
        "translation": "翻译模型",
        "summarization": "摘要生成模型",
        "question-answering": "问答模型",
        "image-classification": "图像分类模型",
        "object-detection": "目标检测模型",
        "default": "AI模型",
    },
}


def get_label(key: str) -> str:
    return LABELS.get(LANG, LABELS["en"]).get(key, key)


def get_model_desc(model_id: str, pipeline_tag: str, tags: list) -> str:
    """Generate short description for a model"""
    # Check if it's an LLM pick with predefined description
    if model_id in LLM_PICKS:
        return LLM_PICKS[model_id].get(LANG, LLM_PICKS[model_id]["en"])
    
    tags_lower = [t.lower() for t in (tags or [])]
    parts = []
    
    # Determine model type from tags and name
    model_lower = model_id.lower()
    
    # Check for specific model types
    if "instruct" in model_lower or "chat" in tags_lower or "conversational" in tags_lower:
        parts.append("Chat/Instruct" if LANG == "en" else "对话/指令")
    elif "base" in model_lower:
        parts.append("Base model" if LANG == "en" else "基座模型")
    
    if "code" in tags_lower or "coder" in model_lower:
        parts.append("Code" if LANG == "en" else "代码")
    
    if "vision" in tags_lower or "vl" in model_lower or "image" in model_lower:
        parts.append("Vision" if LANG == "en" else "视觉")
    
    if "multilingual" in tags_lower or "multi" in model_lower:
        parts.append("Multilingual" if LANG == "en" else "多语言")
    
    # Add quantization info
    quant_found = []
    for q in ["gguf", "gptq", "awq", "exl2"]:
        if q in tags_lower or q in model_lower:
            quant_found.append(q.upper())
    if quant_found:
        parts.append(f"Quant:{','.join(quant_found)}")
    
    # Base description from pipeline_tag
    templates = MODEL_DESC_TEMPLATES.get(LANG, MODEL_DESC_TEMPLATES["en"])
    base = templates.get(pipeline_tag, "")
    
    if parts:
        return ", ".join(parts)
    elif base:
        return base
    return templates.get("default", "AI model")


class RecommendPanel(Static):
    """Recommendation buttons panel"""

    def compose(self) -> ComposeResult:
        yield Label("Quick Browse", classes="panel-title")
        yield Horizontal(
            Button("🔥\nHot", id="rec-trending", variant="error", classes="rec-btn"),
            Button("💬\nLLM", id="rec-llm-type", variant="primary", classes="rec-btn"),
            Button("🔢\nEmbed", id="rec-embed", variant="warning", classes="rec-btn"),
            Button("⭐\nPicks", id="rec-llm", variant="success", classes="rec-btn"),
            classes="rec-row",
        )
        yield Horizontal(
            Button("EN", id="lang-en", variant="primary" if LANG == "en" else "default", classes="lang-btn"),
            Button("中", id="lang-zh", variant="primary" if LANG == "zh" else "default", classes="lang-btn"),
            classes="lang-row",
        )


class DownloadPanel(Static):
    """Download panel"""

    def compose(self) -> ComposeResult:
        yield Label("Download Model", classes="panel-title")
        yield Input(placeholder="Model ID (select from table or enter manually)", id="model-input")
        yield Horizontal(
            Vertical(
                Label("Download Source"),
                Select(SOURCE_OPTIONS, id="source-select", value="hf-mirror"),
            ),
            Vertical(
                Label("Local Directory"),
                Input(placeholder="./models", id="dir-input", value="./models"),
            ),
            classes="filter-row",
        )
        yield Button("Download", id="download-btn", variant="success")
        yield Static("", id="download-status")


class ModelTUI(App):
    """HuggingFace Model Search & Download TUI"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 2;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: auto 1fr;
    }
    
    SearchPanel {
        row-span: 1;
        column-span: 1;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    RecommendPanel {
        row-span: 1;
        column-span: 1;
        border: solid $warning;
        padding: 1;
        margin: 1;
    }
    
    .rec-row {
        width: 100%;
        height: auto;
    }
    
    .rec-btn {
        width: 1fr;
        height: 4;
        margin: 0 1;
        content-align: center middle;
    }
    
    .lang-row {
        width: 100%;
        height: auto;
        margin-top: 1;
    }
    
    .lang-btn {
        width: 1fr;
        margin: 0 1;
    }
    
    DownloadPanel {
        row-span: 1;
        column-span: 1;
        border: solid $success;
        padding: 1;
        margin: 1;
    }
    
    ResultsPanel {
        row-span: 1;
        column-span: 3;
        border: solid $secondary;
        padding: 1;
        margin: 1;
        height: 100%;
    }
    
    .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    .filter-row {
        height: auto;
        margin: 1 0;
    }
    
    .filter-row > Vertical {
        width: 1fr;
        margin-right: 1;
    }
    
    .button-row {
        margin-top: 1;
    }
    
    .button-row Button {
        margin-right: 1;
    }
    
    #results-table {
        height: 100%;
    }
    

    #download-status {
        margin-top: 1;
        color: $text-muted;
    }
    
    Input {
        margin-bottom: 1;
    }
    
    Select {
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "focus_search", "Search"),
        Binding("d", "focus_download", "Download"),
        Binding("r", "refresh", "Refresh"),
    ]

    TITLE = "HuggingFace Model Browser"

    def compose(self) -> ComposeResult:
        yield Header()
        yield SearchPanel()
        yield RecommendPanel()
        yield DownloadPanel()
        yield ResultsPanel()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("#", "Model ID", "Params", "VRAM", "Local", "Downloads", "Description")
        table.cursor_type = "row"
        self.search_models()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_focus_download(self) -> None:
        self.query_one("#model-input", Input).focus()

    def action_refresh(self) -> None:
        self.search_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search-btn":
            self.search_models()
        elif event.button.id == "trending-btn":
            self.query_one("#task-select", Select).value = ""
            self.query_one("#sort-select", Select).value = "downloads"
            self.query_one("#search-input", Input).value = ""
            self.search_models()
        elif event.button.id == "download-btn":
            self.download_model()
        elif event.button.id == "rec-trending":
            self.load_recommend("trending")
        elif event.button.id == "rec-llm-type":
            self.load_recommend("text-generation")
        elif event.button.id == "rec-embed":
            self.load_recommend("embedding")
        elif event.button.id == "rec-llm":
            self.load_recommend("llm")
        elif event.button.id == "lang-en":
            self.switch_language("en")
        elif event.button.id == "lang-zh":
            self.switch_language("zh")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_data = event.data_table.get_row(event.row_key)
        if row_data:
            model_id = row_data[1]
            self.query_one("#model-input", Input).value = model_id

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            self.search_models()
        elif event.input.id == "model-input":
            self.download_model()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("task-select", "sort-select"):
            if event.value is not Select.BLANK:
                self.search_models()

    def switch_language(self, lang: str) -> None:
        global LANG
        LANG = lang
        # Update button styles
        en_btn = self.query_one("#lang-en", Button)
        zh_btn = self.query_one("#lang-zh", Button)
        en_btn.variant = "primary" if lang == "en" else "default"
        zh_btn.variant = "primary" if lang == "zh" else "default"
        # Refresh current view
        self.search_models()

    @work(exclusive=True, thread=True)
    def search_models(self) -> None:
        search = self.query_one("#search-input", Input).value
        task = self.query_one("#task-select", Select).value
        sort = self.query_one("#sort-select", Select).value

        self.call_from_thread(self.update_status, "Searching...")

        try:
            api = HfApi()
            models = list(
                api.list_models(
                    filter=task if task else None,
                    search=search if search else None,
                    sort=sort,
                    direction=-1,
                    limit=30,
                )
            )
            self.call_from_thread(self.update_status, f"Found {len(models)} models, fetching details...")
            
            # Fetch detailed info for each model in parallel
            model_details = {}
            def fetch_detail(m):
                params = get_model_params(api, m.id)
                return m.id, params
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_detail, m): m for m in models}
                for future in as_completed(futures):
                    try:
                        model_id, params = future.result()
                        model_details[model_id] = params
                    except:
                        pass
            
            self.call_from_thread(self.update_table, models, model_details)
            self.call_from_thread(self.update_status, f"Found {len(models)} models")
        except Exception as e:
            self.call_from_thread(self.update_status, f"Error: {e}")

    def update_table(self, models, model_details: dict = None) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()
        model_details = model_details or {}
        
        for i, model in enumerate(models, 1):
            tags = model.tags or []
            pipeline_tag = model.pipeline_tag or ""
            
            # Try to get exact params from API, fallback to name extraction
            params_count = model_details.get(model.id, 0)
            if params_count <= 0:
                _, params_count = extract_params_from_name(model.id, tags)
            
            params_str = format_params(params_count)
            quant = extract_quant_info(model.id, tags)
            vram = estimate_vram(params_count, quant)
            local = can_run_locally(vram)
            desc = get_model_desc(model.id, pipeline_tag, tags)
            
            table.add_row(
                str(i),
                model.id[:35] if len(model.id) > 35 else model.id,
                params_str,
                vram,
                local,
                format_number(model.downloads),
                desc[:30] if len(desc) > 30 else desc,
            )

    def update_status(self, message: str) -> None:
        self.query_one("#download-status", Static).update(message)

    @work(exclusive=True, thread=True)
    def load_recommend(self, category: str) -> None:
        """Load recommended models and display in results table"""
        self.call_from_thread(self.update_status, f"Loading {category} models...")
        
        try:
            api = HfApi()
            from datetime import datetime, timedelta
            
            if category == "llm":
                # LLM curated picks - fetch details for each
                model_details = {}
                def fetch_llm_detail(model_id):
                    try:
                        info = api.model_info(model_id)
                        params = 0
                        if info.safetensors and info.safetensors.total:
                            params = info.safetensors.total
                        return model_id, info, params
                    except:
                        return model_id, None, 0
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(fetch_llm_detail, m) for m in LLM_PICKS.keys()]
                    results = []
                    for future in as_completed(futures):
                        try:
                            model_id, info, params = future.result()
                            if info:
                                results.append((model_id, info, params))
                        except:
                            pass
                
                self.call_from_thread(self.update_results_llm, results)
                self.call_from_thread(self.update_status, f"Showing {len(results)} LLM-picked models")
                return
            
            # Other categories
            if category == "text-generation":
                models = list(api.list_models(filter="text-generation", sort="downloads", direction=-1, limit=20))
            elif category == "embedding":
                models = list(api.list_models(filter="feature-extraction", sort="downloads", direction=-1, limit=20))
            else:  # trending
                models = list(api.list_models(sort="trending_score", direction=-1, limit=20))
            
            # Fetch details in parallel
            model_details = {}
            def fetch_detail(m):
                params = get_model_params(api, m.id)
                return m.id, params
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_detail, m): m for m in models}
                for future in as_completed(futures):
                    try:
                        model_id, params = future.result()
                        model_details[model_id] = params
                    except:
                        pass
            
            self.call_from_thread(self.update_table, models, model_details)
            self.call_from_thread(self.update_status, f"Showing {len(models)} {category} models")
        except Exception as e:
            self.call_from_thread(self.update_status, f"Error: {e}")

    def update_results_llm(self, results) -> None:
        """Update results table with LLM picks"""
        table = self.query_one("#results-table", DataTable)
        table.clear()
        
        for i, (model_id, info, params_count) in enumerate(results, 1):
            tags = info.tags or []
            pipeline_tag = info.pipeline_tag or ""
            if params_count <= 0:
                _, params_count = extract_params_from_name(model_id, tags)
            
            quant = extract_quant_info(model_id, tags)
            vram = estimate_vram(params_count, quant)
            local = can_run_locally(vram)
            desc = get_model_desc(model_id, pipeline_tag, tags)
            
            table.add_row(
                str(i),
                model_id[:35] if len(model_id) > 35 else model_id,
                format_params(params_count),
                vram,
                local,
                format_number(info.downloads or 0),
                desc[:30] if len(desc) > 30 else desc,
            )

    @work(exclusive=True, thread=True)
    def download_model(self) -> None:
        model_id = self.query_one("#model-input", Input).value
        source = self.query_one("#source-select", Select).value
        local_dir = self.query_one("#dir-input", Input).value or "./models"

        if not model_id:
            self.call_from_thread(self.update_status, "Please enter a model ID")
            return

        self.call_from_thread(self.update_status, f"Downloading {model_id}...")

        try:
            if source == "hf-mirror":
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                from huggingface_hub import snapshot_download

                target_dir = os.path.join(local_dir, model_id.replace("/", "_"))
                path = snapshot_download(
                    repo_id=model_id,
                    local_dir=target_dir,
                    resume_download=True,
                )
                self.call_from_thread(self.update_status, f"Downloaded to: {path}")
            else:
                try:
                    from modelscope.hub.snapshot_download import snapshot_download

                    path = snapshot_download(model_id=model_id, cache_dir=local_dir)
                    self.call_from_thread(self.update_status, f"Downloaded to: {path}")
                except ImportError:
                    self.call_from_thread(
                        self.update_status, "ModelScope not installed. Run: pip install modelscope"
                    )
        except Exception as e:
            self.call_from_thread(self.update_status, f"Error: {e}")


def main():
    app = ModelTUI()
    app.run()


if __name__ == "__main__":
    main()
