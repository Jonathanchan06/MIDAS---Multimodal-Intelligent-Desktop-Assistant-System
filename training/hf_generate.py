"""Greedy generation for eval prompts with a HF model (no Ollama, no grammar),
for scoring a fine-tune Ollama cannot load yet. Prompts come from a dump of
`extraction_eval.py --cases all --dump-prompts prompts.json`; grade the output with
`extraction_eval.py --regrade OUT.json` (the MIDAS venv).
usage: hf_generate.py MODEL_DIR_OR_ID prompts.json out.json"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, prompts_path, out_path = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda").eval()
im_end = tok.convert_tokens_to_ids("<|im_end|>")
recs = []
for p in json.load(open(prompts_path, encoding="utf-8")):
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in p["messages"]) + "<|im_start|>assistant\n"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    t = time.perf_counter()
    with torch.no_grad():
        g = model.generate(ids, max_new_tokens=400, do_sample=False, eos_token_id=[im_end, tok.eos_token_id], pad_token_id=tok.eos_token_id)
    raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        args = json.loads(raw)
        failures = []
    except json.JSONDecodeError:
        args, failures = None, [{"category": "invalid_output", "path": "", "message": "not JSON"}]
    recs.append({"tool": p["tool"], "set": p["set"], "text": p["text"], "raw": raw, "args": args,
                 "seconds": time.perf_counter() - t, "output_tokens": int(g.shape[1] - ids.shape[1]),
                 "failures": failures})
json.dump({"model": f"hf:{model_id}", "sampling": {"greedy": True}, "repeats": 1, "cases": "all",
           "eval_now": "2026-11-25T10:00:00", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "summary": {}, "records": recs}, open(out_path, "w", encoding="utf-8"), indent=2)
print("wrote", out_path, len(recs))
