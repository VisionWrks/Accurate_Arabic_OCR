# Iqraa - Arabic OCR & TTS Pipeline

An **ongoing project** for Arabic text processing: **OCR** (Optical Character Recognition) for printed/historical documents, **TTS** (Text-to-Speech) synthesis, and **web crawling** for Arabic text corpora.

---

## Project Structure

```
Iqraa/
├── scripts/
│   ├── ocr/                        # OCR pipeline
│   │   ├── detect_lines.py         # Line detection from scanned pages
│   │   └── archive/                # Earlier experimental OCR scripts
│   │       ├── LLM5.py
│   │       ├── LLM5+gimini.py
│   │       ├── LLM5+groq.py
│   │       └── groq_only.py
│   ├── tts/                        # Text-to-Speech
│   │   ├── f5_tts_infer.py         # F5-TTS Arabic inference
│   │   ├── google_tts.py           # Google TTS
│   │   ├── mms_tts.py              # Meta MMS TTS
│   │   ├── download_voice.py       # Voice model downloader
│   │   └── make_mp3_from_sentences.py  # Batch sentence-to-MP3
│   └── crawlers/                   # Web crawlers for Arabic text
│       ├── shamela_crawl.py        # Shamela library crawler
│       └── tafsir_crawl.py         # Tafsir text crawler
├── .gitignore
└── README.md
```

## Components

### OCR
- Detects text lines from scanned Arabic pages using morphological and projection-based methods.
- Supports deskewing, background removal, and contrast enhancement.
- LLM-assisted post-processing for diacritics and grammar correction (Groq, Gemini).

### TTS
- Arabic speech synthesis using multiple engines: F5-TTS, Google TTS, Meta MMS.
- Batch processing of sentences into MP3 audio files.

### Crawlers
- Scrape Arabic text from online libraries (Shamela, Quran tafsir sources).
- Produce plain-text corpora for TTS and other downstream tasks.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or install packages as needed
```

For OCR, install Tesseract:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ara
# macOS
brew install tesseract tesseract-lang
```

---

## Authors & Contributions
Developed by **Taleb Elm**.
Contributions are welcome!
