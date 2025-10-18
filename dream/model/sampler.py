import sys
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as dists

sys.path.append("..")
from config import Config
from utils.sampling_params import SamplingParams

class Sampler(nn.Module):
    def __init__(self, device, config: Config, sampling_params: SamplingParams):
        super().__init__()
        self.device = device
        self.config = config
        self.sampling_params = sampling_params

        gen_length = self.config.max_new_tokens
        self.block_length = self.config.block_length
        steps = self.config.steps
        assert gen_length % self.block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({self.block_length})"
        num_blocks = gen_length // self.block_length
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        self.steps_per_block = steps // num_blocks

        self.timesteps = torch.linspace(1, self.sampling_params.eps, self.steps_per_block + 1, device=device)

    def top_p_logits(self, logits, top_p=None):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits

    def top_k_logits(self, logits, top_k=None):
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits


    def sample_tokens(self, logits, neg_entropy=False):
        temperature = self.sampling_params.temperature
        top_p = self.sampling_params.top_p
        top_k = self.sampling_params.top_k
        margin_confidence = self.sampling_params.margin_confidence
        neg_entropy = neg_entropy or self.sampling_params.neg_entropy

        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1:
            logits = self.top_p_logits(logits, top_p)
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)

        if temperature > 0:
            try:
                x0 = dists.Categorical(probs=probs).sample()
                confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except:
                confidence, x0 = probs.max(dim=-1)
        else:
            confidence, x0 = probs.max(dim=-1)
        
        if margin_confidence:
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            # Extract top1 and top2 probabilities
            top1_probs = sorted_probs[:, 0] 
            top2_probs = sorted_probs[:, 1] 
            # Calculate confidence as top1 - top2
            confidence = top1_probs - top2_probs 
        
        if neg_entropy:
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            confidence = torch.sum(probs * log_probs, dim=-1)
        
        return confidence, x0

    def forward(self, i, x, logits, mask_index, current_block_start, dual_cache=False):
        current_block_end = current_block_start + self.block_length
        mask_token_id = self.config.mask_token_id
        if self.sampling_params.alg == 'confidence_threshold':
            mask_logits = logits[mask_index]
        
            confidence, x0 = self.sample_tokens(mask_logits)
            
            if dual_cache:
                x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
            else:
                x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
            
            x_[mask_index] = x0.clone()
            full_confidence[mask_index] = confidence
            full_confidence[:, self.block_length:] = -torch.inf
            
            current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
            
            selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
            transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
            
            select_index = select_index.to(x.device)
            transfer_index[0, select_index[0]] = True
            for k in range(1, current_transfer_tokens):
                if selected_confidence[0, k] < self.sampling_params.threshold:
                    transfer_index[0, select_index[0, k]] = False
            if dual_cache:
                x[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
            else:
                x[:, current_block_start:][transfer_index] = x_[transfer_index]
        else:
            if i == self.steps_per_block:
                return x
            t = self.timesteps[i]
            s = self.timesteps[i + 1]
            mask_index[:, self.block_length:] = False
            mask_logits = logits[mask_index]
            confidence, x0 = self.sample_tokens(mask_logits, neg_entropy=True)
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < self.steps_per_block - 1 else int(num_mask_token)
            if dual_cache:
                full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
            else:
                full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            full_confidence[:, self.block_length:] = -torch.inf
            
            if number_transfer_tokens > 0:
                alg_temp = self.sampling_params.alg_temp
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                if dual_cache:
                    x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                else:
                    x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                if dual_cache:
                    x[:, current_block_start:current_block_end][row_indices,transfer_index] = x_[row_indices,transfer_index]
                else:
                    x[:, current_block_start:][row_indices,transfer_index] = x_[row_indices,transfer_index]

        return x