import sys
import types
from transformers import AutoTokenizer

from model.modeling_dream import DreamModel
sys.path.append("..")
from utils.sampling_params import SamplingParams

class LLMEngine:
    def __init__(self, model, device, sampling_params: SamplingParams):
        self.model = model
        self.device = device
        self.sampling_params = sampling_params
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = DreamModel.from_pretrained(model, trust_remote_code=True)
        self.model = self.model.to(device).eval()

        from model.generation_utils_block import DreamGenerationMixin
        self.model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, self.model)
        self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)

    def generate(self, chats, context_max_length=None):
        inputs = self.tokenizer.apply_chat_template(
            chats,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=context_max_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        out = self.model.diffusion_generate(
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

        # decode output ids to text
        attn = inputs["attention_mask"]
        prompt_lens = attn.sum(dim=1).tolist()

        texts = []
        for i, p_len in enumerate(prompt_lens):
            seq = out.sequences[i]
            gen_ids = seq[p_len:]  # strip the prompt tokens
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            # truncate at eos if present
            if self.tokenizer.eos_token and self.tokenizer.eos_token in text:
                text = text.split(self.tokenizer.eos_token)[0]
            # also strip any trailing special tokens residue
            text = text.replace(self.tokenizer.eos_token or "", "").strip()
            texts.append(text)

        return texts
