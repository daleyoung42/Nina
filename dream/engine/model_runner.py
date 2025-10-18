import sys
import torch
from torch.nn import functional as F

sys.path.append("..")
from config import Config
from utils.sampling_params import SamplingParams
from utils.sequence import Sequence
from model.modeling_dream import DreamModel
from model.sampler import Sampler

class ModelRunner:
    def __init__(self, model, device, config: Config, sampling_params: SamplingParams):
        self.config = config
        self.device = device
        # Sampler as a part of the model
        self.sampling_params = sampling_params

        self.model = DreamModel.from_pretrained(model, trust_remote_code=True)
        self.model = self.model.to(device).eval()

        self.sampler = Sampler(device, config, sampling_params).to(device)

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

    def run(self, seqs: list[Sequence], is_prefill: bool):
        # [1, 2], [2, 3] -> [[1, 2], [2, 3]]
        input_ids = torch.stack([seq.token_ids for seq in seqs], dim=0)
        attention_mask = torch.stack([seq.attention_mask for seq in seqs], dim=0)
        # print(f"input_ids: {input_ids}, attention_mask: {attention_mask}")
        current_block_start = seqs[0].num_prompt_tokens + seqs[0].current_block * self.config.block_length
        outputs = self.run_model(input_ids, attention_mask, current_block_start)
        return outputs