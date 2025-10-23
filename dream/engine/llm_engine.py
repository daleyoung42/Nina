import sys
import types
from dataclasses import fields
from transformers import AutoTokenizer


from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner

sys.path.append("..")
from config import Config
from utils.sequence import Sequence
from utils.sampling_params import SamplingParams

import torch

class LLMEngine:
    def __init__(self, model, device, sampling_params: SamplingParams, **kwargs):
        self.model = model
        self.device = device
        self.sampling_params = sampling_params

        config_fields = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model=model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model_runner = ModelRunner(self.model, self.device, self.config, self.sampling_params)

        self.scheduler = Scheduler(self.config, eos=self.tokenizer.eos_token)

    def add_request(self, chat, context_max_length=None):
        need_pad = True
        if "v2" in self.model:
            chat = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(chat, return_tensors="pt")
            need_pad = False
        else:
            inputs = self.tokenizer.apply_chat_template(
                    chat,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=context_max_length,
                )
        input_ids = inputs["input_ids"].squeeze(0).to(self.device)
        attention_mask = inputs["attention_mask"].squeeze(0).to(self.device)

        seq = Sequence(token_ids=input_ids, attention_mask=attention_mask, max_length=self.config.max_new_tokens, mask_token_id=self.config.mask_token_id, need_pad=need_pad)
        # print(f"[LLM] New request: seq_id={seq.seq_id} num_tokens={seq.num_tokens}, input_ids={input_ids}, attention_mask={attention_mask}")
        self.scheduler.add(seq)
        return seq
    
    def step(self):
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        # print(f"[LLM] Scheduled {len(scheduled_seqs)} sequences, is_prefill={is_prefill}")
        # for seq in scheduled_seqs:
            # print(f"  - seq_id={seq.seq_id} num_tokens={seq.num_tokens}")

        outputs = self.model_runner.run(scheduled_seqs, is_prefill)

        finished_flags = self.scheduler.postprocess(scheduled_seqs, outputs)
        # for seq, finished in zip(scheduled_seqs, finished_flags):
            # print(f"[LLM] Finished seq_id={seq.seq_id} finished={finished}")

        outputs = outputs if outputs is not None else []
        return scheduled_seqs, outputs, finished_flags

    def generate(self, chats, context_max_length=None):
        # if "v2" in self.model:
        #     import time
        #     t0 = time.perf_counter()
        #     chats = self.tokenizer.apply_chat_template(
        #         chats,
        #         tokenize=False,
        #         add_generation_prompt=True,
        #     )
        #     model_inputs = self.tokenizer(chats, return_tensors="pt").to(self.device)
        #     generated_ids = self.model_runner.model.generate(
        #         model_inputs["input_ids"],
        #         tokenizer=self.tokenizer,
        #         block_size=32,
        #         max_new_tokens=128,
        #         small_block_size=8,
        #         threshold=0.9,
        #     )
        #     response = [self.tokenizer.decode(generated_id[model_inputs["input_ids"].shape[1]:], skip_special_tokens=True) for generated_id in generated_ids]
        #     print(f"elapsed time: {time.perf_counter() - t0}s")
        #     return response    
        for chat in chats:
            self.add_request(chat, context_max_length=context_max_length)
        gen_outputs = {}
        # if "v2" in self.model:
        #     print("run")
        #     seqs, is_prefill = self.scheduler.schedule()
        #     input_ids = torch.stack([seq.token_ids for seq in seqs], dim=0)
        #     generated_ids = self.model_runner.model.generate(
        #         input_ids,
        #         tokenizer=self.tokenizer,
        #         block_size=32,
        #         max_new_tokens=256,
        #         small_block_size=8,
        #         threshold=0.9,
        #     )
        #     response = [self.tokenizer.decode(generated_id[input_ids.shape[1]:], skip_special_tokens=True) for generated_id in generated_ids]
        #     return response
        
        while not self.scheduler.is_finished():
            scheduled_seqs, outputs, finished_flags = self.step()
            # print(f"[LLM] Scheduled_seqs: {scheduled_seqs} Step outputs: {outputs}, finished_flags: {finished_flags}")
            for seq, output, finished in zip(scheduled_seqs, outputs, finished_flags):
                # print(f"[LLM] Step result: seq_id={seq.seq_id} output={output} finished={finished}")
                if finished:
                    gen_outputs[seq.seq_id] = (output, seq.num_prompt_tokens)

        # print(f"outputs: {gen_outputs}")

        texts = []
        for i in range(len(chats)):
            output, prompt_len = gen_outputs[i]
            gen_ids = output[prompt_len:]  # strip the prompt tokens
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            # truncate at eos if present
            if self.tokenizer.eos_token and self.tokenizer.eos_token in text:
                text = text.split(self.tokenizer.eos_token)[0]
            # also strip any trailing special tokens residue
            text = text.replace(self.tokenizer.eos_token or "", "").strip()
            texts.append(text)

        return texts
