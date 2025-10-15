from dataclasses import dataclass

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    block_length: int = 32
    steps: int = 128
    max_new_tokens: int = 128
    output_history: bool = True
    return_dict_in_generate: bool = True
    mask_token_id: int = 151666
    dual_cache: bool = False