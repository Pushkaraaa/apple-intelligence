import logging
from text_processor.config import ProcessingType, ProcessorConfig
from text_processor.text_processor import TextProcessor

# Custom configuration
config = ProcessorConfig(
    model_name="llama3.2",
    temperature=0.7,
    hotkeys={
        ProcessingType.SUMMARIZE: "<shift>+<ctrl>+j",
        ProcessingType.QNA: "<shift>+<ctrl>+k",
        ProcessingType.REPHRASE: "<shift>+<ctrl>+l"
    }
)

processor = TextProcessor(config)

try:
    print(f"""
Text Processor Started
---------------------
Available commands:
- Ctrl+Shift+J: Summarize clipboard text
- Ctrl+Shift+K: Q&A mode (press once to set context, again with question)
- Ctrl+Shift+L: Rephrase clipboard text
- Press Ctrl+C to exit

Text requirements:
- Minimum length: {config.min_text_length} characters
- Maximum length: {config.max_text_length} characters
""")
    processor.start()
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    processor.stop()