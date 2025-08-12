from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_active_heads: int = 6 # number of heads to use in attention
    activate_heads_after_n_epochs: int = 1000
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    def __setattr__(self, key, value):
        assert key != 'n_active_heads' or value <= self.n_head
        super().__setattr__(key, value)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_active_heads <= self.n_head