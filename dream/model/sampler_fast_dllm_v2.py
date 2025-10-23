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

    def sample_with_top_p(self, logits, top_p=0.95, temperature=1.0):
        # Calculate probabilities
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            p_1t = torch.softmax(logits, dim=-1)
            x_1 = p_1t.argmax(dim=-1)
            return x_1, p_1t
        # Compute softmax probabilities                    
        probs = F.softmax(scaled_logits, dim=-1)
        # Sort the probabilities to perform top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        probs[indices_to_remove] = 0

        # Renormalize so that the probabilities of remaining tokens sum to 1
        # Add a small epsilon value to prevent division by zero
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        normalized_probs = probs / probs_sum

        p_1t = normalized_probs
        x_1 = torch.multinomial(p_1t[0], num_samples=1).unsqueeze(0).squeeze(-1)

        return x_1, p_1t

    def forward(self, x, logits, top_p, temperature, threshold, mask_idx, start, end):
        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
        # Select tokens with probability greater than threshold from p_1t
        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

        unmask_idx = (x1_p > threshold)
        max_prob_idx = x1_p.argmax(dim=-1)
        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
        unmask_idx = unmask_idx & mask_idx[:, start:end]
        x[:, start:end][unmask_idx] = x_1[unmask_idx]
        return x