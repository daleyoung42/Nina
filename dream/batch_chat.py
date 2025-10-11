#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
from transformers import AutoModel, AutoTokenizer
from model.modeling_dream import DreamModel
import types


def decode_batch_outputs(tokenizer, inputs, outputs, eos_token: str):
    """
    Slice generated sequences by each prompt length (sum of attention_mask),
    then decode to text for each sample.
    """
    attn = inputs["attention_mask"]
    prompt_lens = attn.sum(dim=1).tolist()

    texts = []
    for i, p_len in enumerate(prompt_lens):
        seq = outputs.sequences[i]
        gen_ids = seq[p_len:]  # strip the prompt tokens
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        # truncate at eos if present
        if eos_token and eos_token in text:
            text = text.split(eos_token)[0]
        # also strip any trailing special tokens residue
        text = text.replace(tokenizer.eos_token or "", "").strip()
        texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    # parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=128, help="diffusion timesteps")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy",
                        choices=["origin", "maskgit_plus", "topk_margin", "entropy"])
    parser.add_argument("--alg-temp", type=float, default=0.1)
    parser.add_argument("--context-max-length", type=int, default=None,
                        help="optional truncation length for tokenizer.apply_chat_template")
    args = parser.parse_args()

    print(f"[Init] model={args.model} device={args.device}")
    print(f"[Gen ] steps={args.steps} max_new_tokens={args.max_new_tokens} temp={args.temperature} "
          f"top_p={args.top_p} top_k={args.top_k} alg={args.alg} alg_temp={args.alg_temp}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    model = DreamModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = model.to(args.device).eval()

    from model.generation_utils_block import DreamGenerationMixin
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    prompts = [
        "List all primes under 100.",
        "List all primes under 100.",
    ]

    chats = [[{"role": "user", "content": p}] for p in prompts]
    
    print(f"chats: {chats}")

    inputs = tokenizer.apply_chat_template(
            chats,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=args.context_max_length,
        )
    
    print(f"inputs: {inputs}")
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)

    out = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )
        
    print(f"output: {out}")

    # Decode per-sample
    replies = decode_batch_outputs(tokenizer, inputs, out, eos_token=tokenizer.eos_token)

    # Append assistant replies & print nicely
    print()
    for i, r in enumerate(replies):
        print(f"[S{i}] {r}\n")
    print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
