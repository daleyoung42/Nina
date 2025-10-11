import sys
from collections import deque

sys.path.append("..")
from config import Config
from utils.sequence import Sequence, SequenceStatus

class Scheduler:
    def __init__(self, config: Config, eos):
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_seqs = config.max_num_seqs
        self.eos = eos

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running
    
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
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
            num_batched_tokens += next_seq.num_tokens
            num_seqs += 1
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # Decode
        # TODO: there is no decode stage as the scheduled seqs will always be finished for now

        return scheduled_seqs, False
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        length = len(seqs)
        for seq in seqs:
            # print(f"Mark seq_id={seq.seq_id} as finished")
            seq.status = SequenceStatus.FINISHED
            self.running.remove(seq)

        return [True for _ in range(length)]