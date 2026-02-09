from transformers import pipeline
import numpy as np
import soundfile as sf

pipe = pipeline("text-to-speech", model="facebook/mms-tts-ara")

text = "إِنَّ الْحِكْمَةَ فِي وَصَايَا لُقْمَانَ لِابْنِهِ تَظْهَرُ فِي الْأُسْلُوبِ وَالْمَعْنَى."
out = pipe(text)

audio = out["audio"]

# Make it a clean 1-D float32 numpy array
audio = np.array(audio, dtype=np.float32).squeeze()

sr = int(out["sampling_rate"])

sf.write("test.wav", audio, sr, subtype="PCM_16")
print("Wrote test.wav with", len(audio), "samples at", sr, "Hz")

