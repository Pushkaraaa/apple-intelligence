import logging
from text_processor.config import ProcessingType, ProcessorConfig
from text_processor.text_processor import TextProcessor

# Custom configuration
config = ProcessorConfig(
    model_name="llama3.2",
    temperature=0.7,
    hotkeys={
        ProcessingType.SUMMARIZE: "<cmd>+1",
        ProcessingType.QNA: "<cmd>+2",
        ProcessingType.REPHRASE: "<cmd>+3"
    }
)

processor = TextProcessor(config)

try:
    print(f"""
Text Processor Started whats the most unque feature of taj mahal's gardens
---------------------
Available commands:
- {config.hotkeys[ProcessingType.SUMMARIZE]}: Summarize clipboard text
- {config.hotkeys[ProcessingType.QNA]}: Q&A mode (press once to set context, again with question)
- {config.hotkeys[ProcessingType.REPHRASE]}: Rephrase clipboard text
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