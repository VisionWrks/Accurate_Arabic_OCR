import sys
import os
from gtts import gTTS

# 1. Fix for Mishkal Import and Recursion
import mishkal.tashkeel
# Increase recursion depth for complex Arabic grammar analysis
sys.setrecursionlimit(2000)

def process_arabic_file(input_filename, output_audio="output.mp3"):
    try:
        # 2. Initialize the correct Mishkal class
        # Note: In some versions it is TashkeelClass, in others Tashkeeler.
        # This try-except block ensures it works regardless of version.
        try:
            vocalizer = mishkal.tashkeel.TashkeelClass()
        except AttributeError:
            vocalizer = mishkal.tashkeel.Tashkeeler()

        # 3. Read the text file (ensure UTF-8 encoding)
        if not os.path.exists(input_filename):
            print(f"Error: {input_filename} not found.")
            return

        with open(input_filename, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()

        if not raw_text:
            print("The text file is empty.")
            return

        print(f"--- Processing File: {input_filename} ---")
        
        # 4. Apply Tashkeel (Automatic Voweling)
        print("Applying diacritics (Tashkeel)...")
        vocalized_text = vocalizer.tashkeel(raw_text)
        
        # 5. Convert to Speech using gTTS
        print("Generating audio...")
        tts = gTTS(text=vocalized_text, lang='ar', slow=False)
        
        # 6. Save output
        tts.save(output_audio)
        print(f"Done! Audio saved to: {output_audio}")
        print(f"Sample of vocalized text: {vocalized_text[:100]}...")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure you have a file named 'input.txt' in the same folder
    # or change the name here:
    process_arabic_file("input.txt")