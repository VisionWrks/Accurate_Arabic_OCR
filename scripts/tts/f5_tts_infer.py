import os
import torch
import torchaudio
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder

# This specific model uses a character-based tokenizer. 
# We must ensure the vocab is loaded correctly so 'ب' isn't read as English 'b'
def load_arabic_vocab(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        chars = [line.strip() for line in f.readlines()]
    return {char: i for i, char in enumerate(chars)}

def run_arabic_tts():
    ckpt_path = "model_547500_8_18.pt"
    vocab_path = "vocab.txt"
    ref_audio_path = "ref_arabic.wav"
    
    with open("my_tashkeel.txt", "r", encoding="utf-8") as f:
        gen_text = f.read().strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using Device: {device.upper()} ---")

    # 1. Load the Arabic-Specific Vocab
    vocab_map = load_arabic_vocab(vocab_path)
    
    # 2. Architecture Match
    model_params = dict(dim=1024, depth=22, heads=18, ff_mult=2, text_dim=512, conv_layers=8)
    transformer = DiT(**model_params, text_num_embeds=2581, mel_dim=100)
    model = CFM(transformer, vocab_char_map=vocab_map).to(device)
    model = load_checkpoint(model, ckpt_path, device, use_ema=False)
    vocoder = load_vocoder(is_local=False, device=device)

    # 3. Reference Audio Prep
    audio, sr = torchaudio.load(ref_audio_path)
    if sr != 24000:
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
    audio = torch.mean(audio, dim=0, keepdim=True)

    # 4. The Secret Sauce: The Prompt
    # For IbrahimSalah v2, the model needs to 'hear' the reference 
    # and then 'see' the text to continue.
    ref_text = "مرحبا بك كيف يمكنني مساعدتك اليوم" # Transcript of your ref audio
    
    print(f"--- Synthesizing Arabic: {gen_text} ---")

    with torch.no_grad():
        # Using 16 steps for speed, cfg 2.0 for clarity
        generated, _ = model.sample(
            cond=audio.to(device),
            text=[ref_text + " " + gen_text],
            duration=torch.tensor([audio.shape[-1] // 256 + int(len(gen_text) * 18)]).to(device),
            steps=16,
            cfg_strength=2.0
        )

    # 5. Save and Export
    generated_wave = vocoder.decode(generated.transpose(1, 2))
    output_path = "output_arabic_fixed.wav"
    torchaudio.save(output_path, generated_wave.cpu(), 24000)
    print(f"SUCCESS! Check {output_path}")

if __name__ == "__main__":
    run_arabic_tts()