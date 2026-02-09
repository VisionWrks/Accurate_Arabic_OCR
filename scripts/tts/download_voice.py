from huggingface_hub import hf_hub_download

# This is a clean sample from an open-source Arabic dataset
repo_id = "NeoBoy/arabic-tts-wav-24k"
filename = "female_0458.wav" # A high-quality 24kHz sample

print(f"Downloading reference sample: {filename}...")
local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

# Move/link it to your local directory as 'ref_arabic.wav'
import shutil
shutil.copy(local_path, "ref_arabic.wav")
print("Done! File saved as 'ref_arabic.wav'")