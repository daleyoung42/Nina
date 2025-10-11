import sys
import types
import torch

sys.path.append("..")
from config import Config
from utils.sampling_params import SamplingParams
from utils.sequence import Sequence
from model.modeling_dream import DreamModel

class ModelRunner:
    def __init__(self, model, device, config: Config, sampling_params: SamplingParams):
        self.config = config
        # TODO: ModelRunner should be decoupled with SamplingParams
        self.sampling_params = sampling_params

        self.model = DreamModel.from_pretrained(model, trust_remote_code=True)
        self.model = self.model.to(device).eval()

        from model.generation_utils_block import DreamGenerationMixin
        self.model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, self.model)
        self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)

    @torch.inference_mode()
    def run_model(self, input_ids, attention_mask):
        outputs = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.sampling_params.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.sampling_params.steps,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            alg=self.sampling_params.alg,
            alg_temp=self.sampling_params.alg_temp,
        )
        # print(f"Model outputs: {outputs}")
        return outputs
    
    def run(self, seqs: list[Sequence], is_prefill: bool):
        # [1, 2], [2, 3] -> [[1, 2], [2, 3]]
        input_ids = torch.stack([seq.token_ids for seq in seqs], dim=0)
        attention_mask = torch.stack([seq.attention_mask for seq in seqs], dim=0)
        # print(f"input_ids: {input_ids}, attention_mask: {attention_mask}")
        outputs = self.run_model(input_ids, attention_mask)
        return outputs