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

        seq = Sequence(token_ids=input_ids, attention_mask=attention_mask, max_length=self.config.max_new_tokens, mask_token_id=self.config.mask_token_id)
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
        for chat in chats:
            self.add_request(chat, context_max_length=context_max_length)

        gen_outputs = {}
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
