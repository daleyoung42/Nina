import sys
import types
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch.nn import functional as F
from transformers.utils import ModelOutput

sys.path.append("..")
from config import Config
from utils.sampling_params import SamplingParams
from utils.sequence import Sequence
from model.modeling_dream import DreamModel
from model.sampler import Sampler

@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None

class ModelRunner:
    def __init__(self, model, device, config: Config, sampling_params: SamplingParams):
        self.config = config
        self.device = device
        # TODO: ModelRunner should be decoupled with SamplingParams
        self.sampling_params = sampling_params

        self.model = DreamModel.from_pretrained(model, trust_remote_code=True)
        self.model = self.model.to(device).eval()

        from model.generation_utils_block import DreamGenerationMixin
        self.model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, self.model)
        self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)

        self.sampler = Sampler(device, config, sampling_params).to(device)

    @torch.inference_mode()
    def run_model(self, input_ids, attention_mask):

        output_history = self.config.output_history
        return_dict_in_generate = self.config.return_dict_in_generate
        max_length = self.config.max_new_tokens + input_ids.shape[-1]
        mask_token_id = self.config.mask_token_id
        block_length = self.config.block_length
        steps = self.config.steps

        dual_cache = self.config.dual_cache

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, self.sampling_params.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Initialize cache for the prompt
        past_key_values = None

        # Process each block
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            # update cache
            model_output = self.model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            # confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
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
                x = self.sampler(i, x, logits, mask_index, current_block_start, block_length, steps_per_block, timesteps, dual_cache=dual_cache)
                i += 1

                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    break

        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

        # outputs = self.model.diffusion_generate(
        #     self.sampler,
        #     input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=self.sampling_params.max_new_tokens,
        #     output_history=False,
        #     return_dict_in_generate=True,
        #     steps=self.sampling_params.steps,
        #     temperature=self.sampling_params.temperature,
        #     top_p=self.sampling_params.top_p,
        #     top_k=self.sampling_params.top_k,
        #     alg=self.sampling_params.alg,
        #     alg_temp=self.sampling_params.alg_temp,
        # )
        # # print(f"Model outputs: {outputs}")
        # return outputs
    
    def run(self, seqs: list[Sequence], is_prefill: bool):
        # [1, 2], [2, 3] -> [[1, 2], [2, 3]]
        input_ids = torch.stack([seq.token_ids for seq in seqs], dim=0)
        attention_mask = torch.stack([seq.attention_mask for seq in seqs], dim=0)
        # print(f"input_ids: {input_ids}, attention_mask: {attention_mask}")
        outputs = self.run_model(input_ids, attention_mask)
        return outputs