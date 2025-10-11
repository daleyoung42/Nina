from dataclasses import dataclass

@dataclass
class SamplingParams:
    steps: int = 128
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = None
    top_k: int = None
    alg: str = "entropy" # choices=["origin", "maskgit_plus", "topk_margin", "entropy"]
    alg_temp: float = 0.1
    