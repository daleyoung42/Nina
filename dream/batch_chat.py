#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
from utils.sampling_params import SamplingParams
from llm import LLM


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

    sampling_params = SamplingParams(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )
    print(f"[Init] sampling params: {sampling_params}")

    llm = LLM(args.model, args.device, sampling_params)

    prompts = [
        "List all primes under 100.",
        "List all primes under 100.",
    ]

    chats = [[{"role": "user", "content": p}] for p in prompts]

    out = llm.generate(chats, context_max_length=args.context_max_length)
        
    print()
    for i, r in enumerate(out):
        print(f"[S{i}] {r}\n")
    print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
