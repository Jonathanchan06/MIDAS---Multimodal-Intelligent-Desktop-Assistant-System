"""LoRA fine-tune of a small Qwen model for MIDAS argument extraction.

Trains on training/data/extraction_{train,val}.jsonl (from gen_extraction_data.py):
each record is [system, *history, user, assistant] where system is exactly what
build_extraction_messages sends live. Loss is on the assistant JSON (+ <|im_end|>)
only.

Prompts are rendered by hand in ChatML so they match Ollama's qwen2.5 template
byte for byte. lm_head runs only on target positions so the 151k-vocab logits
never materialise for the whole sequence (fits a 6GB laptop GPU).

Runs in its own venv (torch/transformers/peft), not the MIDAS one:
    ~/midas-train-venv/Scripts/python.exe training/train_lora.py
Outputs go outside OneDrive (default ~/midas-train/<run>): adapter/, merged/.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent


def render(messages: list[dict]) -> tuple[str, str]:
    """(prompt, target): prompt ends with the open assistant turn, matching
    Ollama's qwen2.5 template for system + history + user."""
    *ctx, answer = messages
    assert answer["role"] == "assistant"
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in ctx)
    return prompt + "<|im_start|>assistant\n", answer["content"] + "<|im_end|>"


def encode(tok, path: Path) -> list[tuple[list[int], int]]:
    """[(ids, n_prompt)] — loss on ids[n_prompt:]."""
    out = []
    for line in path.open(encoding="utf-8"):
        prompt, target = render(json.loads(line)["messages"])
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        t = tok(target, add_special_tokens=False)["input_ids"]
        out.append((p + t, len(p)))
    return out


def batches(data, batch_tokens: int, shuffle: bool):
    """Length-bucketed batches capped at batch_tokens (padded size)."""
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    groups, cur, cur_max = [], [], 0
    for i in order:
        n = len(data[i][0])
        if cur and max(cur_max, n) * (len(cur) + 1) > batch_tokens:
            groups.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        groups.append(cur)
    if shuffle:
        random.shuffle(groups)
    return groups


def collate(data, idx, pad_id, device):
    L = max(len(data[i][0]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(idx), L), dtype=torch.long)
    lossmask = torch.zeros((len(idx), L), dtype=torch.bool)
    for r, i in enumerate(idx):
        seq, n_prompt = data[i]
        ids[r, :len(seq)] = torch.tensor(seq)
        attn[r, :len(seq)] = 1
        # position j predicts token j+1: mark positions whose NEXT token is target
        lossmask[r, n_prompt - 1:len(seq) - 1] = True
    return ids.to(device), attn.to(device), lossmask.to(device)


def batch_loss(model, ids, attn, lossmask):
    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids, attention_mask=attn).last_hidden_state
    logits = inner.lm_head(hidden[lossmask]).float()
    labels = ids[:, 1:][lossmask[:, :-1]]
    return torch.nn.functional.cross_entropy(logits, labels, reduction="sum"), labels.numel()


@torch.no_grad()
def evaluate(model, data, args, pad_id, device):
    model.eval()
    tot, n, exact = 0.0, 0, 0
    for idx in batches(data, args.batch_tokens, shuffle=False):
        ids, attn, lm = collate(data, idx, pad_id, device)
        loss, k = batch_loss(model, ids, attn, lm)
        tot, n = tot + loss.item(), n + k
        # teacher-forced token accuracy of whole answers (cheap exact-match proxy)
        inner = model.get_base_model()
        hidden = inner.model(input_ids=ids, attention_mask=attn).last_hidden_state
        pred = inner.lm_head(hidden[lm]).argmax(-1)
        gold = ids[:, 1:][lm[:, :-1]]
        counts = lm.sum(1).tolist()
        for p, g in zip(pred.split(counts), gold.split(counts)):
            exact += bool((p == g).all())
    model.train()
    return tot / n, exact / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(Path.home() / "midas-train" / "qwen2.5-0.5b-extract"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-tokens", type=int, default=6144)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train = encode(tok, Path(args.data) / "extraction_train.jsonl")
    val = encode(tok, Path(args.data) / "extraction_val.jsonl")
    lens = [len(s) for s, _ in train]
    print(f"train {len(train)}  val {len(val)}  tokens/ex mean {sum(lens)/len(lens):.0f} max {max(lens)}")

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(batches(train, args.batch_tokens, False)) / args.grad_accum)
    total = int(steps_per_epoch * args.epochs)
    warm = max(1, int(total * args.warmup))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

    vl, ve = evaluate(model, val, args, pad_id, device)
    print(f"step 0/{total}  val_loss {vl:.4f}  val_exact {ve:.1%}")
    best = vl
    step, t0 = 0, time.time()
    model.train()
    log = []
    while step < total:
        groups = batches(train, args.batch_tokens, shuffle=True)
        for g0 in range(0, len(groups), args.grad_accum):
            chunk = groups[g0:g0 + args.grad_accum]
            ntok = sum(sum(len(train[i][0]) - train[i][1] for i in idx) for idx in chunk)
            run = 0.0
            for idx in chunk:
                ids, attn, lm = collate(train, idx, pad_id, device)
                loss, _ = batch_loss(model, ids, attn, lm)
                (loss / ntok).backward()
                run += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"step {step}/{total}  loss {run/ntok:.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
            if step % steps_per_epoch == 0 or step == total:
                vl, ve = evaluate(model, val, args, pad_id, device)
                log.append({"step": step, "val_loss": vl, "val_exact": ve})
                print(f"step {step}/{total}  val_loss {vl:.4f}  val_exact {ve:.1%}", flush=True)
                if vl <= best:
                    best = vl
                    model.save_pretrained(out / "adapter")
            if step >= total:
                break

    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))
    print(f"done in {time.time()-t0:.0f}s, best val_loss {best:.4f}")
    if args.no_merge:
        return
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    print("merged ->", out / "merged")


if __name__ == "__main__":
    main()
