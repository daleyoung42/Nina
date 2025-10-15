from dataclasses import dataclass

@dataclass
class SamplingParams:
    steps: int = 128
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = None
    top_k: int = None
    margin_confidence: bool = False
    neg_entropy: bool = False
    alg: str = "entropy" # choices=["origin", "confidence_threshold", "entropy"]
    alg_temp: float = 0.1
    eps: float = 1e-3
    threshold: float = 0.9
    