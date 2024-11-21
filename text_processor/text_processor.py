import os
import time
import logging
import pyperclip
from pynput import keyboard
from typing import Optional, Dict, Callable
from langchain.chat_models.ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from .config import ProcessingType, ProcessorConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('text_processor.log')
    ]
)

class TextProcessor:
    """Main class for processing clipboard text in various ways."""
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        self._setup_llm()
        self._setup_prompts()
        self.last_processed_text = None
        self.context_text = None  # Stores text for QnA
        self.is_running = False
        self.qa_mode = False

    def _setup_llm(self) -> None:
        """Initialize the language model."""
        try:
            self.llm = ChatOllama(
                model=self.config.model_name,
                base_url="http://localhost:11434",
                temperature=self.config.temperature,
            )
            
            # Set up different chains for different tasks
            self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")
            
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _setup_prompts(self) -> None:
        """Initialize prompt templates for different tasks."""
        self.prompts = {
            ProcessingType.SUMMARIZE: PromptTemplate(
                input_variables=["text"],
                template="Provide a concise summary of the following text:\n\n{text}\n\nSummary:"
            ),
            ProcessingType.REPHRASE: PromptTemplate(
                input_variables=["text"],
                template="Rephrase the following text in a clear and professional way without losing information:\n\n{text}\n\nRephrased version:"
            )
        }

    def _get_clipboard_text(self) -> Optional[str]:
        """Safely get text from clipboard with retries."""
        for attempt in range(self.config.retry_attempts):
            try:
                text = pyperclip.paste()
                if text:
                    return text
            except Exception as e:
                logging.warning(f"Clipboard access failed (attempt {attempt + 1}): {str(e)}")
                time.sleep(self.config.retry_delay)
        return None

    def _validate_text(self, text: str, allow_short: bool = False) -> bool:
        """Validate the clipboard text."""
        if not text:
            return False
        if not allow_short and len(text.strip()) < self.config.min_text_length:
            logging.info("Text too short for processing")
            return False
        if len(text) > self.config.max_text_length:
            logging.info("Text exceeds maximum length")
            return False
        return True

    def process_text(self, proc_type: ProcessingType) -> Optional[str]:
        """Process text based on the specified type."""
        try:
            text = self._get_clipboard_text()
            if not self._validate_text(text):
                return None

            logging.info(f"Processing text with {proc_type.value}...")
            
            if proc_type == ProcessingType.QNA:
                self.context_text = text
                self.qa_mode = True
                print("\nContext set for Q&A. Please copy your question to clipboard and press the Q&A hotkey again.")
                return None
            
            chain = LLMChain(llm=self.llm, prompt=self.prompts[proc_type])
            result = chain.invoke({"text": text})["text"]

            if result:
                pyperclip.copy(result)
                logging.info(f"{proc_type.value} result copied to clipboard")
                print(f"\n{proc_type.value.capitalize()} Result:", result)
                return result

        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
        return None

    def handle_qa(self) -> Optional[str]:
        """Handle Q&A mode."""
        try:
            if not self.context_text:
                print("Please set context first by copying text and pressing Q&A hotkey.")
                return None

            question = self._get_clipboard_text()
            if not self._validate_text(question, allow_short=True):
                return None

            logging.info("Processing Q&A...")
            
            # Create a proper Document object for the context
            doc = Document(page_content=self.context_text)
            
            # Run the chain with properly formatted input
            result = self.qa_chain.run(
                input_documents=[doc],
                question=question
            )

            if result:
                pyperclip.copy(result)
                logging.info("Q&A result copied to clipboard")
                print("\nQuestion:", question)
                print("Answer:", result)
                return result

        except Exception as e:
            logging.error(f"Error in Q&A: {str(e)}")
            print(f"\nError processing Q&A: {str(e)}")
        return None

    def on_hotkey_activate(self, proc_type: ProcessingType) -> None:
        """Handle hotkey activation."""
        logging.info(f"Hotkey activated for {proc_type.value}")
        if proc_type == ProcessingType.QNA and self.qa_mode:
            self.handle_qa()
        else:
            self.qa_mode = False
            self.process_text(proc_type)

    def start(self) -> None:
        """Start the keyboard listener."""
        try:
            self.is_running = True
            hotkey_map = {
                self.config.hotkeys[proc_type]: lambda p=proc_type: self.on_hotkey_activate(p)
                for proc_type in ProcessingType
            }
            
            self.listener = keyboard.GlobalHotKeys(hotkey_map)
            self.listener.start()
            
            logging.info("Text processor started...")
            
            while self.is_running:
                time.sleep(0.1)
                
        except Exception as e:
            logging.error(f"Error in keyboard listener: {str(e)}")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self.is_running = False
        if hasattr(self, 'listener'):
            self.listener.stop()
            self.listener.join()
        logging.info("Text processor stopped")