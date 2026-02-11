"""Test script for the tashkeel (diacritization) model."""

import sys
import pickle
import torch
import torch.nn as nn
import pyarabic.araby as araby

# --- Model constants (must match training) ---
MAX_LEN = 256
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 8

DIACRITICS = [
    '', araby.FATHA, araby.DAMMA, araby.KASRA, araby.SUKUN,
    araby.FATHATAN, araby.DAMMATAN, araby.KASRATAN,
    araby.SHADDA,
    araby.SHADDA + araby.FATHA, araby.SHADDA + araby.DAMMA,
    araby.SHADDA + araby.KASRA, araby.SHADDA + araby.FATHATAN,
    araby.SHADDA + araby.DAMMATAN, araby.SHADDA + araby.KASRATAN
]
id_to_class = {i: d for i, d in enumerate(DIACRITICS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerDiacritizer(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_LEN, D_MODEL))
        encoder_layers = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=N_LAYERS)
        self.fc_out = nn.Linear(D_MODEL, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return self.fc_out(x)


def diacritize(model, text, char_to_id):
    model.eval()
    clean_text = araby.strip_tashkeel(text)
    input_ids = torch.tensor([[char_to_id.get(c, char_to_id['<UNK>']) for c in clean_text]]).to(DEVICE)

    with torch.no_grad():
        preds = model(input_ids)
        pred_classes = torch.argmax(preds, dim=-1).squeeze(0).cpu().numpy()

    result = ""
    for char, class_id in zip(clean_text, pred_classes):
        result += char + id_to_class[class_id]
    return result


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "best_tashkeel_model.pth"
    vocab_path = sys.argv[2] if len(sys.argv) > 2 else "vocab.pkl"

    # Load vocabulary
    with open(vocab_path, "rb") as f:
        char_to_id = pickle.load(f)
    print(f"Loaded vocab ({len(char_to_id)} chars) from {vocab_path}")

    # Load model
    model = TransformerDiacritizer(len(char_to_id), len(DIACRITICS)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f"Loaded model from {model_path}")
    print(f"Device: {DEVICE}\n")

    # Interactive loop
    print("Enter Arabic text to diacritize (Ctrl+C to quit):")
    print("=" * 60)
    while True:
        try:
            text = input("\n> ").strip()
            if not text:
                continue
            result = diacritize(model, text, char_to_id)
            print(f"  {result}")
        except KeyboardInterrupt:
            print("\n")
            break


if __name__ == "__main__":
    main()
