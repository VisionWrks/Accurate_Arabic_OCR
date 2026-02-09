import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# =========================
# CONFIG
# =========================
SURAH_NUMBER = 2   # change to 1–114
URL = f"https://www.quran-for-all.com/tafsir/al-jalalayn/{SURAH_NUMBER}"

OUT_DIR = Path("out_jalalayn")
OUT_DIR.mkdir(exist_ok=True)

DEBUG = True
PRINT_CHARS = 800
# =========================


# Arabic letters (exclude digits)
ARABIC_LETTERS_RE = re.compile(
    r"[\u0621-\u064A\u066E-\u066F\u0671-\u06D3\u06FA-\u06FF\u0750-\u077F]"
)

# Qur’anic symbols / decorations
QURAN_SYMBOLS_RE = re.compile(r"[﴿﴾۝۞۩]")

# Verse references like [البقرة: 1]
BRACKET_REF_RE = re.compile(r"\[[^\]]+\]")

# Arabic tashkeel to KEEP
TASHKEEL = {
    "\u064B", "\u064C", "\u064D",
    "\u064E", "\u064F", "\u0650",
    "\u0651", "\u0652", "\u0670"
}


# =========================
# CLEANING
# =========================
def remove_boxes_keep_tashkeel(text: str) -> str:
    out = []

    for ch in text:
        cp = ord(ch)

        if ch == "\n":
            out.append(ch)
            continue

        # Remove control chars
        if unicodedata.category(ch).startswith("C"):
            continue

        # Remove Arabic Extended-A (Qur’anic marks → boxes)
        if 0x08A0 <= cp <= 0x08FF:
            continue

        # Remove Arabic Presentation Forms
        if (0xFB50 <= cp <= 0xFDFF) or (0xFE70 <= cp <= 0xFEFF):
            continue

        # Remove tatweel
        if cp == 0x0640:
            continue

        # Keep tashkeel
        if ch in TASHKEEL:
            out.append(ch)
            continue

        # Keep Arabic letters
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            out.append(ch)
            continue

        # Keep punctuation
        if ch in " .,:;،؛!؟()[]{}\"'«»-":
            out.append(ch)
            continue

        # Drop everything else (this removes box glyphs)
        continue

    cleaned = "".join(out)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# =========================
# HEURISTICS
# =========================
def looks_like_quran_line(line: str) -> bool:
    if QURAN_SYMBOLS_RE.search(line):
        return True
    if BRACKET_REF_RE.search(line):
        return True
    return False


# =========================
# EXTRACTION
# =========================
def extract_tafsir_only(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    candidates = soup.select("main, article, section, div")

    blocks = []
    for el in candidates:
        txt = el.get_text("\n", strip=True)
        if len(txt) < 150:
            continue
        if not ARABIC_LETTERS_RE.search(txt):
            continue
        score = len(ARABIC_LETTERS_RE.findall(txt))
        blocks.append((score, txt))

    if not blocks:
        return ""

    blocks.sort(key=lambda x: x[0], reverse=True)
    merged = "\n\n".join(b[1] for b in blocks[:2])

    out_lines = []
    for ln in merged.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if not ARABIC_LETTERS_RE.search(ln):
            continue
        if looks_like_quran_line(ln):
            continue

        ln = remove_boxes_keep_tashkeel(ln)
        if len(ln) >= 12:
            out_lines.append(ln)

    # de-duplicate
    seen = set()
    final = []
    for ln in out_lines:
        if ln not in seen:
            seen.add(ln)
            final.append(ln)

    return "\n".join(final).strip()


# =========================
# MAIN
# =========================
def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,900")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    try:
        driver.get(URL)

        WebDriverWait(driver, 20).until(
            lambda d: ARABIC_LETTERS_RE.search(
                d.find_element(By.TAG_NAME, "body").text or ""
            )
        )

        html = driver.page_source
        (OUT_DIR / f"surah_{SURAH_NUMBER}_snapshot.html").write_text(
            html, encoding="utf-8"
        )

        tafsir = extract_tafsir_only(html)

        if DEBUG:
            print("\n=== TAFSIR OUTPUT (sample) ===\n")
            print(tafsir[:PRINT_CHARS])

        out_file = OUT_DIR / f"jalalayn_surah_{SURAH_NUMBER}_tafsir.txt"
        out_file.write_text(tafsir + "\n", encoding="utf-8")

        print(f"\n✔ Saved: {out_file} ({len(tafsir)} chars)")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
