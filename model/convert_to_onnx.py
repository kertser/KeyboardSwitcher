# Convert the trained v2 model (BiLSTM + masked pooling, PAD=0) to ONNX.
#
# The ONNX graph takes a single int64 tensor [batch, seq] and outputs logits
# [batch, 4].  The sequence axis is dynamic so the C++/Python inference may pad
# to any length (PAD=0 is masked internally) — padding to 45 or to the actual
# word length gives identical results.

import os
import torch

from Languages_torch import load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert_model(model, char_to_index, max_length, device,
                  onnx_path="lang_model.onnx"):
    model.eval()
    dummy = torch.zeros(1, 16, dtype=torch.long, device=device)
    torch.onnx.export(
        model, dummy, os.path.join(SCRIPT_DIR, onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported {onnx_path}")


if __name__ == "__main__":
    args = load_model()
    convert_model(*args)
