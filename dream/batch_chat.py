#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from utils.sampling_params import SamplingParams
from llm import LLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="/disk1/models/hub/Dream-v0-Instruct-7B")
    parser.add_argument("--model", type=str, default="/disk1/models/hub/Fast_dLLM_v2_7B")
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
        "Explain the theory of relativity in simple"
    ]
    chats = [[{"role": "user", "content": p}] for p in prompts]
    # print(f"[Test] Warming up models...")
    # out = llm.generate(chats, context_max_length=args.context_max_length)
    # 计时开始
    t0 = time.perf_counter()
    out = llm.generate(chats, context_max_length=args.context_max_length)
    elapsed = time.perf_counter() - t0

    print()
    for i, r in enumerate(out):
        print(f"[S{i}] {r}\n")
    print("-" * 80 + "\n")

    # 统计 out 的新生成 token 数
    def count_tokens(texts):
        total = 0
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        for s in texts:
            if tok is not None:
                try:
                    # 优先使用 encode 接口
                    total += len(tok.encode(s))
                except Exception:
                    try:
                        # 兼容 tokenizer(s).input_ids 风格
                        total += len(tok(s).input_ids)
                    except Exception:
                        # 最后退化为按空格切分的近似
                        total += len(s.split())
            else:
                total += len(s.split())
        return total

    new_tokens = count_tokens(out)
    tps = new_tokens / elapsed if elapsed > 0 else float("inf")

    print(f"tokens={new_tokens} elapsed={elapsed:.3f}s tps={tps:.2f}")

if __name__ == "__main__":
    main()
