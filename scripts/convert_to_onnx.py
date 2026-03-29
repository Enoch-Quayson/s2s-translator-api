"""
convert_to_onnx.py - Convert all S2S models to ONNX format
"""
import os
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path("./models_cache")
ONNX_DIR = Path("./models_onnx")

def convert_vits(name, src):
    import torch
    from transformers import VitsModel, AutoTokenizer
    out = ONNX_DIR / "tts" / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"  Converting {name}...")
    model = VitsModel.from_pretrained(str(src))
    tokenizer = AutoTokenizer.from_pretrained(str(src))
   
    # Export to ONNX manually
    dummy_input = tokenizer("Hello world", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
   
    torch.onnx.export(
        model,
        (input_ids,),
        str(out / "model.onnx"),
        input_names=["input_ids"],
        output_names=["waveform"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        opset_version=15,
    )
    tokenizer.save_pretrained(str(out))
    model.config.save_pretrained(str(out))
    log(f"  Done: {out}")

def log(msg):
    print(msg)

def log(msg):
    print(msg)

if __name__ == "__main__":
    print("Starting ONNX conversion...")

    import torch
    from transformers import VitsModel, AutoTokenizer

    model_name = "facebook/mms-tts-eng"

    out = ONNX_DIR / "tts" / "mms-tts-eng"
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading model...")
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Converting to ONNX...")
    dummy_input = tokenizer("Hello world", return_tensors="pt")
    input_ids = dummy_input["input_ids"]

    torch.onnx.export(
        model,
        (input_ids,),
        str(out / "model.onnx"),
        input_names=["input_ids"],
        output_names=["waveform"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        opset_version=15,
    )

    tokenizer.save_pretrained(str(out))
    model.config.save_pretrained(str(out))

    print(f"Done! Saved to: {out}")