# Text Summarizer Bot

A Python-based text summarization tool that uses Hugging Face's BART model to generate concise summaries of long articles and documents.

## Features

- Summarizes text using state-of-the-art BART model
- Handles documents up to 10,000 words
- Multiple summary length options (short, medium, long)
- Interactive CLI mode
- Web interface using Gradio
- Language detection
- Progress indicators
- File input/output support

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

1. Basic usage with a text file:
```bash
python summarizer_bot.py --input article.txt --length medium --output summary.txt
```

2. Interactive mode:
```bash
python summarizer_bot.py
```

3. Launch web interface:
```bash
python summarizer_bot.py --web
```

### Options

- `--input`: Text to summarize or file path
- `--length`: Summary length (short, medium, long)
- `--output`: Output filename (default: summary.txt)
- `--web`: Launch web interface

## Example

Input text:
```
The quick brown fox jumps over the lazy dog. The fox was very quick and agile. The dog was quite lazy and didn't move much. This is a famous pangram that contains every letter of the English alphabet.
```

Command:
```bash
python summarizer_bot.py --input example.txt --length short
```

Output:
```
A quick fox jumps over a lazy dog in a famous pangram containing all English alphabet letters.
```

## Notes

- For best results, use English text under 10,000 words
- The model works best with well-structured text
- Processing time depends on text length and available hardware
- GPU acceleration is recommended for longer texts

## License

MIT License 