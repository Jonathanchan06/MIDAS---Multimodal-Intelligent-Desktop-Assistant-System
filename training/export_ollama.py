"""Register a merged fine-tune (train_lora.py's merged/ dir) as an Ollama model.

Copies the chat template from the stock base model so live prompts render
exactly as in training, then `ollama create` imports + quantizes the safetensors.

    python training/export_ollama.py ~/midas-train/qwen2.5-0.5b-extract/merged midas-extract:0.5b

DOES NOT WORK on Ollama 0.40: its safetensors import is MLX-only (no
Qwen2) and Modelfile ADAPTER is gone. Convert to GGUF with llama.cpp's
convert_hf_to_gguf.py first and point FROM at the .gguf (see CLAUDE.md).
"""
import argparse
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("merged", type=Path)
ap.add_argument("name")
ap.add_argument("--template-from", default="qwen2.5:0.5b")
ap.add_argument("--quantize", default="q8_0", help="'' to keep bf16")
args = ap.parse_args()

merged = args.merged.expanduser().resolve()
template = subprocess.run(["ollama", "show", args.template_from, "--template"],
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
modelfile = merged / "Modelfile"
modelfile.write_text(f'FROM {merged}\nTEMPLATE """{template}"""\n', encoding="utf-8")
cmd = ["ollama", "create", args.name, "-f", str(modelfile)]
if args.quantize:
    cmd += ["-q", args.quantize]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
