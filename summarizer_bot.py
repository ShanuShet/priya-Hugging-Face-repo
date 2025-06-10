#!/usr/bin/env python3
import os
import re
import torch
import argparse
import gradio as gr
from tqdm import tqdm
from typing import List, Tuple, Optional
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from langdetect import detect, LangDetectException

class TextSummarizer:
    def __init__(self):
        """Initialize the Hugging Face BART summarization pipeline."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def check_language(self, text: str) -> bool:
        """Check if the text is in English."""
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return True  # Default to True if detection fails

    def chunk_text(self, text: str, max_tokens: int = 1024) -> List[str]:
        """Split text into chunks while preserving sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current_chunk, current_length = [], [], 0

        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if current_length + len(tokens) > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
            
            current_chunk.append(sentence)
            current_length += len(tokens)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _calculate_lengths(self, text: str, length_style: str) -> Tuple[int, int]:
        """Dynamically determine min/max summarization lengths."""
        word_count = len(text.split())

        length_profiles = {
            "short": (0.45, 0.7),    # 450-700 words
            "medium": (1.5, 4.0),   # 1500-4000 words
            "long": (5.5, 8.0)      # 5500-8000 words
        }
        min_factor, max_factor = length_profiles.get(length_style, (0.25, 0.5))

        max_len = max(50, min(300, int(word_count * max_factor)))
        min_len = max(10, min(max_len - 10, int(word_count * min_factor)))

        return min_len, max_len

    def summarize(self, text: str, length_style: str = "medium") -> str:
        """Generate summary with auto-adjusted length constraints."""
        # Check word count
        word_count = len(text.split())
        if word_count > 10000:
            print(f"Warning: Input text exceeds 10,000 words ({word_count} words). Processing anyway...")

        # Check language
        if not self.check_language(text):
            print("Warning: Text may not be in English. Results may be suboptimal.")

        min_len, max_len = self._calculate_lengths(text, length_style)
        chunks = self.chunk_text(text)

        summaries = []
        for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
            try:
                result = self.summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False, truncation=True)
                summaries.append(result[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing chunk {i+1}: {e}")
                summaries.append(chunk)

        return " ".join(summaries)

def process_input(input_text: str) -> Optional[str]:
    """Handle both direct text and file inputs."""
    if os.path.exists(input_text):
        try:
            with open(input_text, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    return input_text

def interactive_mode():
    """Run the summarizer in interactive mode."""
    print("\n=== Text Summarizer Interactive Mode ===")
    print("Choose input method:")
    print("1. Enter text directly")
    print("2. Provide file path")
    
    choice = input("Enter choice (1-2) [default: 1]: ").strip()
    
    if choice == "2":
        file_path = input("\nEnter file path: ").strip()
        text = process_input(file_path)
        if not text:
            print("Error: Could not read file. Exiting...")
            return
    else:
        print("\nEnter your text (type 'END' on a new line to finish):")
        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        text = '\n'.join(lines)
        if not text.strip():
            print("No text provided. Exiting...")
            return

    print("\nSelect summary length:")
    print("1. Short (450-700 words)")
    print("2. Medium (1500-4000 words)")
    print("3. Long (5500-8000 words)")
    
    length_choice = input("Enter choice (1-3) [default: 2]: ").strip()
    length_map = {"1": "short", "2": "medium", "3": "long"}
    length_style = length_map.get(length_choice, "medium")

    summarizer = TextSummarizer()
    summary = summarizer.summarize(text, length_style)

    output_file = input("\nEnter output filename [default: summary.txt]: ").strip() or "summary.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n✅ Summary saved to {output_file}")

    print("\n=== Generated Summary ===")
    print(summary)

def main():
    parser = argparse.ArgumentParser(description="Summarize text using Hugging Face Transformers")
    parser.add_argument("--input", type=str, help="Text to summarize or file path")
    parser.add_argument("--length", choices=["short", "medium", "long"], default="medium", help="Summary length option")
    parser.add_argument("--output", type=str, default="summary.txt", help="Output filename")
    parser.add_argument("--web", action="store_true", help="Launch web interface")

    args = parser.parse_args()

    if args.web:
        iface.launch(share=True)
        return

    if not args.input:
        interactive_mode()
        return

    summarizer = TextSummarizer()
    text = process_input(args.input)
    if not text:
        print("Error: No valid input provided.")
        return

    summary = summarizer.summarize(text, args.length)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n✅ Summary saved to {args.output}")

    print("\n=== Generated Summary ===")
    print(summary)

# Gradio Web UI Interface
def gradio_summarizer(file, text, length_style):
    summarizer = TextSummarizer()
    if file is not None:
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    elif text is not None and text.strip():
        pass  # use the text box input
    else:
        return "Please upload a text file or enter text."
    return summarizer.summarize(text, length_style)

iface = gr.Interface(
    fn=gradio_summarizer,
    inputs=[
        gr.File(label="Upload Text File", file_types=[".txt"]),
        gr.Textbox(label="Or Enter Text Here", lines=10),
        gr.Radio(["short", "medium", "long"], label="Summary Length", value="medium")
    ],
    outputs=gr.Textbox(label="Generated Summary", lines=10),
    title="Hugging Face Text Summarizer",
    description="Upload a text file or enter text to summarize. For best results, use English text under 10,000 words."
)

if __name__ == "__main__":
    main()
