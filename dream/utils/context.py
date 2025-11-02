from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False

    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, block_tables=block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()