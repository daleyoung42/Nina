from dataclasses import dataclass

@dataclass
class Config:
    model: str
    model_name: str = "v2"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    block_length: int = 128
    steps: int = 128
    max_new_tokens: int = 128
    output_history: bool = True
    return_dict_in_generate: bool = True
    mask_token_id: int = 151665
    dual_cache: bool = False
    
    # v2
    small_block_size: int = 8
    stop_token: int = 151645
    use_block_cache = False
    top_p = 0.95
    temperature = 0
    threshold = 0.9