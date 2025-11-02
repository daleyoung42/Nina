import sys
from collections import deque

from engine.block_manager import KVBlockManager

sys.path.append("..")
from config import Config
from utils.sequence import Sequence, SequenceStatus

class Scheduler:
    def __init__(self, config: Config, eos):
        self.config = config
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_seqs = config.max_num_seqs
        self.eos = eos
        self.block_size = config.block_length
        self.stop_token = config.stop_token
        self.max_new_tokens = config.max_new_tokens
        self.mask_id = config.mask_token_id
        self.kv_block_manager = KVBlockManager(
            num_kv_blocks=config.num_kv_blocks,
            kv_block_size=config.kv_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running
    
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        ## TODO: the current schedule policy should ensure every sequence in the batch is processing the same block
        # Prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            next_seq = self.waiting[0]
            if num_batched_tokens + next_seq.num_tokens > self.max_num_batched_tokens:
                break
            next_seq.status = SequenceStatus.RUNNING
            scheduled_seqs.append(self.waiting.popleft())
            self.running.append(next_seq)
            self.kv_block_manager.allocate(next_seq)
            num_batched_tokens += next_seq.num_tokens
            num_seqs += 1
        if scheduled_seqs:
            return scheduled_seqs, True
        # if "v2" in self.config.model_name:
        #     return scheduled_seqs, True
        # Decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            self.kv_block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            num_seqs += 1

        assert scheduled_seqs, "No sequences scheduled!"
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        finished_flags = []
        for seq, token_id in zip(seqs, token_ids):
            seq.token_ids = token_id
            # Update num_tokens directly, could have some masked tokens at the end
            seq.num_tokens = len(token_id)

            is_finished = False
            stop_idx = (seq.token_ids == self.stop_token).nonzero(as_tuple=True)[0]
            if len(stop_idx) > 0 and stop_idx[0] >= seq.num_prompt_tokens:
                is_finished = True
            if len(seq.token_ids) > self.max_new_tokens:
                is_finished = True
            if is_finished:
                finished_flags.append(True)
                seq.status = SequenceStatus.FINISHED
                self.kv_block_manager.deallocate(seq)
                self.running.remove(seq)
            else:
                finished_flags.append(False)

        return finished_flags