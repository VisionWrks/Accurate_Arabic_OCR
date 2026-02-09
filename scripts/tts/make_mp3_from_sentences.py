# -*- coding: utf-8 -*-
"""
make_mp3_from_sentences.py

Reads:
  - JSON: {"sentences": [...]}  (Arabic text, no tashkeel)
  - or TXT: one sentence per line

and synthesizes all sentences into one MP3 using Microsoft Edge TTS.

Usage examples:
  python make_mp3_from_sentences.py \
      --json sentences_split_preserve_chars_newline.json \
      --out output.mp3 \
      --voice ar-SA-HamedNeural --rate +0% --pitch +0Hz --group-size 80

  python make_mp3_from_sentences.py \
      --txt sentences.txt \
      --out output.mp3

Requirements:
  pip install edge-tts
"""

import argparse
import asyncio
import json
import edge_tts


def split_long_lines(lines, max_words=100):
    """
    Take an iterable of lines/sentences.
    For each line:
      - if <= max_words: keep as is
      - if  > max_words: split into two:
            first max_words words,
            and the remaining words as a second sentence.
    """
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        words = line.split()
        if len(words) <= max_words:
            result.append(" ".join(words))
        else:
            first = " ".join(words[:max_words])
            rest = " ".join(words[max_words:])
            result.append(first)
            result.append(rest)
    return result


async def synthesize(sentences, out_file, voice, rate, pitch, group_size, sep, verbose):
    """Stream TTS from text batches and append to a single MP3."""
    # Ensure output file is empty
    open(out_file, "wb").close()

    total = len(sentences)
    for i in range(0, total, group_size):
        group = sentences[i:i + group_size]
        text = (sep or ".\n\n").join(group)
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)

        with open(out_file, "ab") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])

        if verbose:
            print(f"✔ Processed group {i//group_size + 1}/{(total-1)//group_size + 1} ({len(group)} sentences)")

    print(f"\n✅ Done! Saved {total} sentences to '{out_file}'")


async def main():
    parser = argparse.ArgumentParser(
        description="Read Arabic sentences (JSON or TXT) and synthesize to MP3."
    )

    # Mutually exclusive input options: either JSON or TXT
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--json",
        help="Input JSON file with {'sentences': [...]}"
    )
    input_group.add_argument(
        "--txt",
        help="Input plain text file (one sentence per line)"
    )

    parser.add_argument("--out", required=True, help="Output MP3 filename")
    parser.add_argument("--voice", default="ar-SA-HamedNeural", help="Voice (default: ar-SA-HamedNeural)")
    parser.add_argument("--rate", default="+0%", help="Speech rate (e.g. +10% or -5%)")
    parser.add_argument("--pitch", default="+0Hz", help="Speech pitch (e.g. +0Hz or -2Hz)")
    parser.add_argument("--group-size", type=int, default=80, help="Number of sentences per TTS request")
    parser.add_argument("--sep", default=".\n\n", help="Separator between sentences in each batch")
    parser.add_argument("--verbose", action="store_true", help="Print progress messages")
    args = parser.parse_args()

    raw_sentences = []

    if args.json:
        # Load sentences from JSON
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_sentences = [s.strip() for s in data.get("sentences", []) if isinstance(s, str)]
    elif args.txt:
        # Load sentences from plain text file, line by line
        with open(args.txt, "r", encoding="utf-8") as f:
            raw_sentences = [line.rstrip("\n") for line in f]

    # Apply line splitting rule: if >5 words, cut into two (5 + rest)
    sentences = split_long_lines(raw_sentences, max_words=100)

    # Final cleanup (strip empties)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        raise SystemExit("❌ No sentences found in the input file.")

    await synthesize(
        sentences,
        args.out,
        args.voice,
        args.rate,
        args.pitch,
        args.group_size,
        args.sep,
        args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
