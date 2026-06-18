# This library us a language toolset for the project.

# import the necessary packages
import torch
import torch.nn as nn
import pickle
import json
import os

russian_layout = 'ёйцукенгшщзхъфывапролджэ\ячсмитьбю.ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭ/ЯЧСМИТЬБЮ,'
english_layout = '''`qwertyuiop[]asdfghjkl;'\zxcvbnm,./~QWERTYUIOP{}ASDFGHJKL:"|ZXCVBNM<>?'''
hebrew_layout = ";/'קראטוןםפ][שדגכעיחלךף,\זסבהנמצתץ."
special_characters = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
numbers = "0123456789"

PAD_IDX = 0  # reserved; no real character is assigned index 0

class LanguageClassifier(nn.Module):
    """BiLSTM character classifier with masked mean+max pooling.

    num_embeddings = max(char_to_index.values()) + 1  (real chars 1..N + PAD@0)
    """

    def __init__(self, num_embeddings, embed_size=64, hidden_size=128,
                 num_layers=2, num_classes=4, dropout=0.3, pad_idx=PAD_IDX):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(num_embeddings, embed_size,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # *2 for bidirectional, *2 for mean+max concat
        self.fc = nn.Linear(hidden_size * 2 * 2, num_classes)

    def forward(self, x):
        # x: [batch, seq] int64 (may arrive as [batch, 1, seq] from the dataset
        # shape — flatten the singleton dim defensively).
        x = x.view(x.size(0), -1)
        mask = (x != self.pad_idx).unsqueeze(-1).float()        # [B, T, 1]

        emb = self.embedding(x)                                 # [B, T, E]
        out, _ = self.lstm(emb)                                 # [B, T, 2H]
        out = out * mask                                        # zero PAD steps

        lengths = mask.sum(dim=1).clamp(min=1.0)                # [B, 1]
        mean_pool = out.sum(dim=1) / lengths                    # [B, 2H]

        neg_inf = torch.finfo(out.dtype).min
        out_masked = out.masked_fill(mask == 0, neg_inf)
        max_pool = out_masked.max(dim=1).values                 # [B, 2H]
        # Guard the all-PAD edge case (should not happen for valid input).
        max_pool = torch.where(torch.isfinite(max_pool),
                               max_pool, torch.zeros_like(max_pool))

        feat = torch.cat([mean_pool, max_pool], dim=-1)         # [B, 4H]
        return self.fc(self.dropout(feat))


# Loading the model
def _load_char_to_index(directory):
    """Prefer dictionary.json (shared with the C++ build), fall back to .pkl."""
    json_path = os.path.join(directory, "dictionary.json")
    pkl_path = os.path.join(directory, "dictionary.pkl")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: int(v) for k, v in raw.items()}
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_model(directory=None,
               embed_size=64, hidden_size=128, num_layers=2,
               num_classes=4, weights="lang_model.pth", device=None):
    directory = directory or os.path.dirname(os.path.realpath(__file__))
    char_to_index = _load_char_to_index(directory)
    num_embeddings = max(char_to_index.values()) + 1
    if device is None:
        device = torch.device("cpu")

    model = LanguageClassifier(num_embeddings, embed_size, hidden_size,
                               num_layers, num_classes)
    state = torch.load(os.path.join(directory, weights), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    max_length = 64  # train-time cap; masking makes inference length-agnostic
    return [model, char_to_index, max_length, device]


# AI assisted language detection - hebrew,english,russian
def predict_language(text, model, char_to_index, max_length, device):
    indices = [char_to_index.get(c, PAD_IDX) for c in text if c in char_to_index]
    if not indices:
        return None
    if len(indices) < max_length:
        indices += [PAD_IDX] * (max_length - len(indices))
    indices = indices[:max_length]

    tensor = torch.tensor(indices, dtype=torch.long).view(1, -1).to(device)
    with torch.no_grad():
        logits = model(tensor)
    cls = int(torch.argmax(logits, dim=1).item())
    return {0: None, 1: "en", 2: "he", 3: "ru"}.get(cls)


# Helper function to create a conversion map
def create_conversion_map(source_layout, target_layout):
    return {ord(s): ord(t) for s, t in zip(source_layout, target_layout)}


# Helper function to convert text
def convert_text(text, conversion_map):
    return text.translate(conversion_map)


# Main function to convert text from one layout to another
def convert_text_bidirectional(text, from_layout, to_layout):
    if to_layout == hebrew_layout:
        text = text.lower()
    return convert_text(text, create_conversion_map(from_layout, to_layout))


if __name__ == "__main__":
    args = load_model()
    print("Predicted:", predict_language("привет", *args))
