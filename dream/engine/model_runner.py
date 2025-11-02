import sys
import torch
from torch.nn import functional as F

sys.path.append("..")
from config import Config
from utils.sampling_params import SamplingParams
from utils.sequence import Sequence
from model.modeling_dream import DreamModel
from model.modeling_fast_dllm_v2 import Fast_dLLM_QwenForCausalLM
from model.sampler import Sampler
from model.sampler_fast_dllm_v2 import Sampler as Sampler_v2
from transformers import AutoTokenizer

class ModelRunner:
    def __init__(self, model, device, config: Config, sampling_params: SamplingParams):
        self.config = config
        self.device = device

        self.world_size = 1

        # Sampler as a part of the model
        self.sampling_params = sampling_params
        # temp for v2 debug
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.config.hf_config.torch_dtype)
        torch.set_default_device(self.device)
        if "v2" not in self.model_name:
            self.model = DreamModel.from_pretrained(model, torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model = self.model.eval()
        if "v2" not in self.model_name:
            self.sampler = Sampler(device, config, sampling_params).to(device)
        else:
            self.sampler = Sampler_v2(device, config, sampling_params).to(device)

        self.allocate_kv_cache()

        torch.set_default_dtype(default_dtype)
        torch.set_default_device('cpu')

    
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")

        block_bytes = 2 * hf_config.num_hidden_layers * self.config.kv_block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kv_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kv_blocks > 0

        print(f"[ModelRunner] GPU memory total: {total / (1024**3):.2f} GB, used: {used / (1024**3):.2f} GB, peak: {peak / (1024**3):.2f} GB, current allocated: {current / (1024**3):.2f} GB")
        print(f"[ModelRunner] Each kv_cache block size: {block_bytes / (1024**2):.2f} MB")
        print(f"[ModelRunner] Allocating kv_cache with {config.num_kv_blocks} blocks, each block has size {self.config.kv_block_size}, total size: {block_bytes * config.num_kv_blocks / (1024**3):.2f} GB")

        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kv_blocks, self.config.kv_block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            # print(f"Module: {module}")
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    @torch.inference_mode()
    def run_model(self, input_ids, attention_mask, current_block_start):

        max_length = self.config.max_new_tokens
        mask_token_id = self.config.mask_token_id
        block_length = self.config.block_length

        dual_cache = self.config.dual_cache

        # pad input_ids to max_length
        # x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        x = input_ids.clone()
        # print(f"input_ids shape: {input_ids.shape}")
        # print(f"max_length: {max_length}")
        # print(f"x shape: {x.shape}")
        
        # Handle block configuration
        if block_length is None:
            block_length = max_length  # Default: single block (original behavior)

        # TODO: post init attention mask in Sequence class
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # print(f"padding attention_mask")
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            # print(f"full attention_mask")
            tok_idx = None
            attention_mask = "full"

        # Process each block
        # for num_block in range(num_blocks):
            # print(f"x before block {num_block}: {x}")
            
        # print(f"x before generation block, current_block_start={current_block_start}: {x}")
        current_block_end = current_block_start + block_length

        # update cache
        model_output = self.model(x, attention_mask, tok_idx, use_cache=True)
        past_key_values = model_output.past_key_values
        logits = model_output.logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        confidence, x0 = self.sampler.sample_tokens(logits)
        x[:, current_block_start] = x0[:, current_block_start]
        
        # Extract only previous block cache
        if not dual_cache:
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values
        else:
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, current_block_start:current_block_end] = 1
            
        i = 1
        while True:
            # Use cache for generation
            if dual_cache:
                mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
            else:
                mask_index = (x[:, current_block_start:] == mask_token_id)
            
            # Prepare attention mask for cached generation
            if attention_mask != "full":
                # Adjust attention mask for current position
                current_attention_mask = attention_mask[:, :, :, current_block_start:]
            else:
                current_attention_mask = attention_mask
            
            if dual_cache:
                model_output = self.model(x[:, current_block_start:current_block_end], current_attention_mask, 
                                tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None, 
                                past_key_values=past_key_values, use_cache=True, dual_cache=dual_cache, replace_position=replace_position)
            else:
                model_output = self.model(x[:, current_block_start:], current_attention_mask, 
                                tok_idx[:, current_block_start:] if tok_idx is not None else None, 
                                past_key_values=past_key_values, use_cache=True)
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            x = self.sampler(i, x, logits, mask_index, current_block_start, dual_cache=dual_cache)
            # print(f"After sampler step {i}, x: {x}")
            i += 1

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break

        return x
    
    def check_small_block_idx(self, x_t, block_size, small_block_size, mask_id):
            """
            在 x_t 的最后一个 block（长度 block_size）里，返回最左边仍含有 mask_id 的 small block 的索引。
            批量情况下：只要 batch 中任意样本在该 small block 有 mask，就选择该 small block。
            若整个最后一个 block 都没有 mask，返回 -1。
            依赖外层的 mask_id 变量。
            """
            tail = x_t[:, -block_size:]                     # [B, block_size]
            mask_idx = (tail == mask_id)                    # [B, block_size], bool
            if not mask_idx.any():
                return -1

            # 常规：要求能被整除；若不能整除也做兜底处理
            full = block_size // small_block_size
            rem = block_size % small_block_size

            for i in range(full):
                s = i * small_block_size
                e = s + small_block_size
                if mask_idx[:, s:e].any():
                    return i

            return -1


    # sampler-step grained
    @torch.inference_mode()
    def run_model_v2(self, input_ids):    
        block_length = self.config.block_length
        block_size = block_length
        num_blocks = self.config.max_new_tokens // block_length
        small_block_size = self.config.small_block_size
        stop_token = self.config.stop_token
        mask_id = self.config.mask_token_id
        use_block_cache = self.config.use_block_cache
        original_input_length = input_ids.shape[1]
        top_p = self.config.top_p
        temperature = self.config.temperature
        threshold = self.config.threshold
        # print(f"block size: {block_size}, num_blocks: {num_blocks}, small_block_size: {small_block_size}")
        # Prefill & Get past_key_values
        print(f"input_ids shape: {input_ids.shape}")
        if input_ids.shape[1] > block_size:
            output = self.model.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
            print(f"forward with past_key_values, input_ids shape: {input_ids[:, :(input_ids.shape[1] // block_size * block_size)].shape}")
            logits, past_key_values = output.logits, output.past_key_values
            if input_ids.shape[1] % block_size == 0:
                mask_idx = (input_ids[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        if stop_token in input_ids[:, original_input_length:]:
            return input_ids
        
        prompt_length = input_ids.shape[1]
        # Initialize x_init with mask_id
        if prompt_length % block_size != 0:
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length % block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
            x_t = x_init.clone()
        else:
            x_t = input_ids.clone()
        block_past_key_values = None
        if stop_token in x_t[:, prompt_length:]:
            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                return x_t
            
        small_block_idx = self.check_small_block_idx(x_t, block_size, small_block_size, mask_id)
        # print(f"small block idx to decode: {small_block_idx}")
        assert small_block_idx != -1, "No small block to decode!"
        # decode small blocks step by step
        small_block_start_idx = small_block_idx * small_block_size
        small_block_end_idx = small_block_start_idx + small_block_size
        start = -block_size + small_block_start_idx
        end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

        mask_idx = (x_t[:, -block_size:] == mask_id)
        if use_block_cache:
            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                output = self.model.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                print(f"forward with block_cache, input_ids shape: {x_t[:, -block_size:].shape}")
                logits, block_past_key_values = output.logits, output.block_past_key_values
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                logits = logits[:, start:end]
            else:
                logits = self.model.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                print(f"2forward with block_cache, input_ids shape: {x_t[:, start:end].shape}")
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        else:
            logits = self.model.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
            print(f"forward without block_cache, input_ids shape: {x_t[:, -block_size:].shape}")
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
            logits = logits[:, start:end]
        x_t = self.sampler.forward(x_t, logits, top_p, temperature, threshold, mask_idx, start, end)
        
        mask_idx = (x_t[:, -block_size:] == mask_id)
        # Decode a complete block, update cache, and generate the next token
        if mask_idx.sum() == 0:
            output = self.model.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
            print(f"2forward with past_key_values, input_ids shape: {x_t[:, -block_size:].shape}")
            logits, past_key_values = output.logits, output.past_key_values
            next_token = logits[:, -1:, :].argmax(dim=-1)
            x_t = torch.cat([x_t, next_token], dim=1)
        input_ids = x_t
        # print(f"input ids shape {input_ids.shape}")
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
        return input_ids
    
    @torch.inference_mode()
    def run_model_v2_block_grained(self, input_ids):
        # outputs = self.model.generate(
        #     input_ids,
        #     tokenizer=self.tokenizer,
        #     block_size=32,
        #     max_new_tokens=128,
        #     small_block_size=8,
        #     threshold=0.9,
        # )
        # return outputs    
        block_length = 32
        block_size = 32
        num_blocks = self.config.max_new_tokens // block_length
        small_block_size = self.config.small_block_size
        stop_token = self.config.stop_token
        mask_id = self.config.mask_token_id
        use_block_cache = self.config.use_block_cache
        original_input_length = input_ids.shape[1]
        top_p = self.config.top_p
        temperature = self.config.temperature
        threshold = self.config.threshold
        # print(f"block size: {block_size}, num_blocks: {num_blocks}, small_block_size: {small_block_size}")
        # Prefill & Get past_key_values
        if input_ids.shape[1] > block_size:
            output = self.model.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None
        
        num_small_blocks = block_size // small_block_size
    
        
        # decode block by block
        for _ in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]
            # Initialize x_init with mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length % block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
            
            x_t = x_init.clone()
            block_past_key_values = None
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate the next token
                if mask_idx.sum() == 0:
                    output = self.model.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    break
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                break

                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                output = self.model.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.model.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.model.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                        # x_1, p_1t = self.model.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        # # Select tokens with probability greater than threshold from p_1t
                        # x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        # x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        # unmask_idx = (x1_p > threshold)
                        # max_prob_idx = x1_p.argmax(dim=-1)
                        # unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        # unmask_idx = unmask_idx & mask_idx[:, start:end]

                        # x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                        x_t = self.sampler.forward(x_t, logits, top_p, temperature, threshold, mask_idx, start, end)

            input_ids = x_t
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
        return input_ids

    def run(self, seqs: list[Sequence], is_prefill: bool):
        # [1, 2], [2, 3] -> [[1, 2], [2, 3]]
        input_ids = torch.stack([seq.token_ids for seq in seqs], dim=0)
        if "v2" in self.model_name:
            outputs = self.run_model_v2(input_ids)
            return outputs

        attention_mask = torch.stack([seq.attention_mask for seq in seqs], dim=0)
        # print(f"input_ids: {input_ids}, attention_mask: {attention_mask}")
        current_block_start = seqs[0].num_prompt_tokens + seqs[0].current_block * self.config.block_length
        outputs = self.run_model(input_ids, attention_mask, current_block_start)
        return outputs