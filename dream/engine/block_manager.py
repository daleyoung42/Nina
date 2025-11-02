import sys
from collections import deque
import xxhash
import numpy as np

sys.path.append("..")
from utils.sequence import Sequence


class KVBlock:

    def __init__(self, kv_block_id):
        self.kv_block_id = kv_block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class KVBlockManager:

    def __init__(self, num_kv_blocks: int, kv_block_size: int):
        self.kv_block_size = kv_block_size
        self.kv_blocks: list[KVBlock] = [KVBlock(i) for i in range(num_kv_blocks)]
        self.hash_to_kv_block_id: dict[int, int] = dict()
        self.free_kv_block_ids: deque[int] = deque(range(num_kv_blocks))
        self.used_kv_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, kv_block_id: int) -> KVBlock:
        kv_block = self.kv_blocks[kv_block_id]
        assert kv_block.ref_count == 0
        kv_block.reset()
        self.free_kv_block_ids.remove(kv_block_id)
        self.used_kv_block_ids.add(kv_block_id)
        return self.kv_blocks[kv_block_id]

    def _deallocate_block(self, kv_block_id: int) -> KVBlock:
        assert self.kv_blocks[kv_block_id].ref_count == 0
        self.used_kv_block_ids.remove(kv_block_id)
        self.free_kv_block_ids.append(kv_block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_kv_block_ids) >= seq.num_kv_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_kv_blocks):
            token_ids = seq.block(i).tolist()
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.kv_block_size else -1
            block_id = self.hash_to_kv_block_id.get(h, -1)
            if block_id == -1 or self.kv_blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_kv_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.kv_block_size
                if block_id in self.used_kv_block_ids:
                    block = self.kv_blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_kv_block_id[h] = block_id
            seq.block_table.append(block_id)
            print(f"[KVBlockManager] Allocated block_id={block_id} for seq_id={seq.seq_id} block_index={i} hash={h} ref_count={block.ref_count}")

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.kv_blocks[block_id]
            block.ref_count -= 1
            print(f"[KVBlockManager] Deallocated block_id={block_id} for seq_id={seq.seq_id} ref_count={block.ref_count}")
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # def can_append(self, seq: Sequence) -> bool:
    #     return len(self.free_kv_block_ids) >= (len(seq) % self.kv_block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.kv_blocks[block_table[-1]]
        if len(seq) % self.kv_block_size == 1:
            # TODO: this strategy is tailored for fast-dllm v2, as it will add a new token when the last block is completed
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_kv_blocks-1).tolist()
            prefix = self.kv_blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            block_id = self.hash_to_kv_block_id.get(h, -1)
            if block_id != -1 and self.kv_blocks[block_id].token_ids == token_ids:
                # Cache hit
                # Free the last block and update the block table
                seq.num_cached_tokens += self.kv_block_size
                last_block.ref_count -= 1
                if last_block.ref_count == 0:
                    self._deallocate_block(last_block.kv_block_id)

                last_block = self.kv_blocks[block_id]
                last_block.ref_count += 1
                block_table[-1] = block_id
                print(f"[KVBlockManager] Cache hit for seq_id={seq.seq_id} hash={h} block_id={block_id}")
            else:
                last_block.update(h, token_ids)
                self.hash_to_kv_block_id[h] = last_block.kv_block_id
                print(f"[KVBlockManager] Appended to block_id={last_block.kv_block_id} for seq_id={seq.seq_id} hash={h}")

            # Append a new block
            new_block_id = self.free_kv_block_ids[0]
            self._allocate_block(new_block_id)
            block_table.append(new_block_id)
        else:
            assert last_block.hash == -1