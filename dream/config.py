from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str
    model_name: str = "v2"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    block_length: int = 32
    steps: int = 128
    max_new_tokens: int = 128
    output_history: bool = True
    return_dict_in_generate: bool = True
    mask_token_id: int = 151665
    dual_cache: bool = False
    kv_block_size: int = 32
    num_kv_blocks: int = -1
    hf_config: AutoConfig | None = None
    gpu_memory_utilization: float = 0.9
    
    # v2
    small_block_size: int = 8
    stop_token: int = 151645
    use_block_cache = True
    top_p = 0.95
    temperature = 0
    threshold = 0.9

    def __post_init__(self):
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)