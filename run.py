import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
# your model import
from model import Transformer
import argparse

# --- stateless helpers (no globals) ---
def pad_to_seq_len(x, seq_len, pad_id):
    return x + [pad_id] * (seq_len - len(x)) if len(x) < seq_len else x[:seq_len]

def make_attention_mask(seq, pad_id):
    return [0 if t == pad_id else 1 for t in seq]

def span_corrupt(token_ids, noise_density, mean_span_len, sentinel_start):
    n_tokens = len(token_ids)
    n_mask = max(1, int(n_tokens * noise_density))
    span_starts = np.random.choice(range(n_tokens), size=n_mask, replace=False)
    span_starts.sort()

    enc, dec = [], []
    cursor, sid = 0, sentinel_start
    for start in span_starts:
        if start < cursor: 
            continue
        span_len = max(1, min(np.random.poisson(mean_span_len), n_tokens - start))
        enc.extend(token_ids[cursor:start]); enc.append(sid)
        dec.append(sid); dec.extend(token_ids[start:start+span_len])
        cursor = start + span_len
        sid -= 1
    enc.extend(token_ids[cursor:])
    dec.append(sid)
    return enc, dec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config_path = args.config
    # ----- config -----
    with open(config_path) as f:
        cfg = json.load(f)
    os.environ["HF_TOKEN"] = cfg.get("api_key","")

    SEQ_LEN = cfg["max_length"]

    # ----- tokenizer (inside main) -----
    MODEL = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    special_tokens = [f"<extra_id_{i}>" for i in range(100)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    print(1)

    PAD_ID = tokenizer.pad_token_id
    BOS_ID = tokenizer.bos_token_id
    EOS_ID = tokenizer.eos_token_id
    sentinel_start = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    # ----- mapping fn (captures tokenizer + IDs) -----
    def chunk_and_pack_for_distill_span(examples):
        all_ids = sum(tokenizer(examples["text"], add_special_tokens=False)["input_ids"], [])
        chunks = [all_ids[i:i+SEQ_LEN] for i in range(0, len(all_ids)-SEQ_LEN, SEQ_LEN)]

        enc_ids, dec_in_ids, labels = [], [], []
        enc_mask, dec_mask = [], []

        for c in chunks:
            enc, dec = span_corrupt(c, noise_density=0.15, mean_span_len=3, sentinel_start=sentinel_start)
            enc_p  = pad_to_seq_len(enc, SEQ_LEN, PAD_ID)
            dec_in = pad_to_seq_len([BOS_ID] + dec[:-1], SEQ_LEN, PAD_ID)
            lab    = pad_to_seq_len(dec + [EOS_ID], SEQ_LEN, -100)

            enc_ids.append(enc_p);      dec_in_ids.append(dec_in);  labels.append(lab)
            enc_mask.append(make_attention_mask(enc_p, PAD_ID))
            dec_mask.append(make_attention_mask(dec_in, PAD_ID))

        return {
            "input_ids": enc_ids,
            "decoder_input_ids": dec_in_ids,
            "labels": labels,
            "attention_mask": enc_mask,
            "decoder_attention_mask": dec_mask,
        }

    
    # ----- data (no multiproc map; keep workers 0) -----
    train_ds_raw = load_dataset("json", data_files=cfg["data_path"], split="train")
    n = len(train_ds_raw); val_n = max(1, int(n * 0.01))
    val_ds_raw = train_ds_raw.select(range(val_n))
    train_ds_raw = train_ds_raw.select(range(val_n, n))

    train_ds = train_ds_raw.map(chunk_and_pack_for_distill_span, batched=True, remove_columns=["text"])
    val_ds   = val_ds_raw.map(chunk_and_pack_for_distill_span, batched=True, remove_columns=["text"])
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    # ----- model -----
    model = Transformer(
        vocab_size=len(tokenizer),
        dim=cfg["dim"],
        encoder_layers=cfg["encoder_layers"],
        decoder_layers=cfg["decoder_layers"],
        num_heads=cfg["num_heads"],
        max_length=cfg["max_length"],
        latent_dim=cfg["latent_dim"]
    )

    # ----- trainer args (no tpu_num_cores) -----
    args_hf = Seq2SeqTrainingArguments(
        output_dir="./rex_output",
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        save_strategy="steps",
        report_to=[],
        save_steps=cfg["eval_steps"],
        logging_steps=cfg["eval_steps"],
        eval_steps=cfg["eval_steps"],
        save_total_limit=3,
        fp16=True,                             # or False if debugging first
        torch_compile=True,
        torch_compile_backend="inductor",
        torch_compile_mode="max-autotune",
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        warmup_steps=cfg["warmup_steps"],
        num_train_epochs=1,
        dataloader_num_workers=0
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

if __name__ == "__main__":
    main()


