from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = "./logs/llmframe.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5 MB per file
                backupCount=3,
                encoding="utf-8",
            ),
        ],
        force=True,
    )

    # Silence noisy third-party libraries
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"llmframe.{name}")
