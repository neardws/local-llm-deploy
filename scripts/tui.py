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
    (r"(\d+\.?\d*)b", 1e9),
    (r"(\d+\.?\d*)m", 1e6),
    (r"(\d+)x(\d+\.?\d*)b", lambda m: int(m.group(1)) * float(m.group(2)) * 1e9),
]

LOCAL_VRAM_GB = 24  # Default GPU VRAM, can be adjusted


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


def extract_params_from_name(model_id: str, tags: list) -> str:
    """Extract parameter count from model name or tags"""
    text = model_id.lower() + " " + " ".join(tags or []).lower()
    for pattern, multiplier in SIZE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            if callable(multiplier):
                params = multiplier(match)
            else:
                params = float(match.group(1)) * multiplier
            if params >= 1e9:
                return f"{params / 1e9:.1f}B"
            elif params >= 1e6:
                return f"{params / 1e6:.0f}M"
    return "N/A"


def extract_quant_info(model_id: str, tags: list) -> str:
    """Extract quantization info from model name or tags"""
    text = model_id.lower() + " " + " ".join(tags or []).lower()
    quants = []
    for q in QUANT_PATTERNS:
        if q in text:
            quants.append(q.upper())
    return ", ".join(quants) if quants else "No"


def estimate_vram(params_str: str, quant_info: str) -> str:
    """Estimate VRAM requirement based on params and quantization"""
    if params_str == "N/A":
        return "N/A"
    
    try:
        if "B" in params_str:
            params = float(params_str.replace("B", "")) * 1e9
        elif "M" in params_str:
            params = float(params_str.replace("M", "")) * 1e6
        else:
            return "N/A"
    except:
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
        grid-size: 2 2;
        grid-columns: 1fr 2fr;
        grid-rows: auto 1fr;
    }
    
    SearchPanel {
        row-span: 1;
        column-span: 1;
        border: solid $primary;
        padding: 1;
        margin: 1;
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
        column-span: 2;
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
        yield DownloadPanel()
        yield ResultsPanel()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("#", "Model ID", "Params", "VRAM", "Quant", "Local", "Downloads", "Likes")
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

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = self.query_one("#results-table", DataTable)
        row_data = table.get_row(event.row_key)
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
                    limit=50,
                )
            )
            self.call_from_thread(self.update_table, models)
            self.call_from_thread(self.update_status, f"Found {len(models)} models")
        except Exception as e:
            self.call_from_thread(self.update_status, f"Error: {e}")

    def update_table(self, models) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()
        for i, model in enumerate(models, 1):
            tags = model.tags or []
            params = extract_params_from_name(model.id, tags)
            quant = extract_quant_info(model.id, tags)
            vram = estimate_vram(params, quant)
            local = can_run_locally(vram)
            table.add_row(
                str(i),
                model.id[:40] if len(model.id) > 40 else model.id,
                params,
                vram,
                quant[:10] if len(quant) > 10 else quant,
                local,
                format_number(model.downloads),
                format_number(model.likes),
            )

    def update_status(self, message: str) -> None:
        self.query_one("#download-status", Static).update(message)

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
