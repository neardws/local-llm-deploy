#!/usr/bin/env python3
"""Search models from HuggingFace Hub"""

import argparse
from huggingface_hub import HfApi

TASK_ALIASES = {
    "embedding": "feature-extraction",
    "embed": "feature-extraction",
    "llm": "text-generation",
    "chat": "text-generation",
    "image": "image-classification",
    "asr": "automatic-speech-recognition",
    "speech": "automatic-speech-recognition",
    "tts": "text-to-speech",
    "translation": "translation",
    "summarization": "summarization",
    "qa": "question-answering",
}


def format_number(num: int) -> str:
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def search_models(
    task: str = None,
    search: str = None,
    sort: str = "downloads",
    limit: int = 20,
):
    api = HfApi()
    
    resolved_task = TASK_ALIASES.get(task, task) if task else None
    
    models = api.list_models(
        filter=resolved_task,
        search=search,
        sort=sort,
        direction=-1,
        limit=limit,
    )
    
    return list(models)


def print_models(models, title: str = "Search Results"):
    if not models:
        print("No models found.")
        return
    
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'#':<4} {'Model ID':<45} {'Downloads':<12} {'Likes':<8}")
    print("-" * 80)
    
    for i, model in enumerate(models, 1):
        model_id = model.id[:44] if len(model.id) > 44 else model.id
        downloads = format_number(model.downloads or 0)
        likes = format_number(model.likes or 0)
        print(f"{i:<4} {model_id:<45} {downloads:<12} {likes:<8}")
    
    print("-" * 80)
    print(f"Total: {len(models)} models\n")


def main():
    parser = argparse.ArgumentParser(
        description="Search models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task embedding --limit 10
  %(prog)s --task llm --sort likes
  %(prog)s --search "qwen"
  %(prog)s --trending

Task aliases:
  embedding, embed -> feature-extraction
  llm, chat        -> text-generation
  image            -> image-classification
  asr, speech      -> automatic-speech-recognition
  tts              -> text-to-speech
        """,
    )
    parser.add_argument(
        "--task", "-t",
        help="Filter by task type (e.g., embedding, llm, image, asr)",
    )
    parser.add_argument(
        "--search", "-s",
        help="Search by keyword (model name, author, etc.)",
    )
    parser.add_argument(
        "--sort",
        choices=["downloads", "likes", "created", "modified"],
        default="downloads",
        help="Sort by field (default: downloads)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Number of results to show (default: 20)",
    )
    parser.add_argument(
        "--trending",
        action="store_true",
        help="Show trending models (sort by downloads)",
    )
    
    args = parser.parse_args()
    
    if args.trending:
        args.sort = "downloads"
    
    title = "HuggingFace Models"
    if args.task:
        resolved = TASK_ALIASES.get(args.task, args.task)
        title = f"HuggingFace Models - Task: {resolved}"
    if args.search:
        title = f"HuggingFace Models - Search: '{args.search}'"
    
    print(f"Searching HuggingFace Hub...")
    models = search_models(
        task=args.task,
        search=args.search,
        sort=args.sort,
        limit=args.limit,
    )
    print_models(models, title)


if __name__ == "__main__":
    main()
