from enum import Enum
from dataclasses import dataclass
from typing import Dict

class ProcessingType(Enum):
    """Enum for different types of text processing."""
    SUMMARIZE = "summarize"
    QNA = "qna"
    REPHRASE = "rephrase"

@dataclass
class ProcessorConfig:
    """Configuration for the text processor."""
    model_name: str = "llama3.2"
    temperature: float = 0.7
    hotkeys: Dict[ProcessingType, str] = None
    min_text_length: int = 20
    max_text_length: int = 8000
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.hotkeys is None:
            self.hotkeys = {
                ProcessingType.SUMMARIZE: "<shift>+<ctrl>+j",  # Summarize
                ProcessingType.QNA: "<shift>+<ctrl>+k",        # Q&A
                ProcessingType.REPHRASE: "<shift>+<ctrl>+l"    # Rephrase
            }