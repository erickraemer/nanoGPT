"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext

import numpy as np
import tabulate
import torch
from omegaconf import OmegaConf
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from nanoGPT import util
from nanoGPT.decoder_block import DecoderBlock
from nanoGPT.gpt import GPT
from nanoGPT.gpt_config import GPTConfig

if len(sys.argv) < 2:
    raise RuntimeError("No config.yaml provided")

arg = sys.argv[1]

if '=' in arg:
    raise RuntimeError("Please provide the path to your config.yaml file as argument")

cfg: GPTConfig = GPTConfig.load(arg)

gpus = util.get_gpus()
util.print_gpus(gpus)
least_busy_gpu: util.NvidiaGPU = min(gpus, key=lambda gpu: gpu.rank())
print(f"Using {least_busy_gpu.name} ({least_busy_gpu.id})\n")
device = f'cuda:{least_busy_gpu.id}' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=cfg.ddp.backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert cfg.data.gradient_accumulation_steps % ddp_world_size == 0
    cfg.data.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

print(f"DDP: {ddp}")

tokens_per_iter = cfg.data.gradient_accumulation_steps * ddp_world_size * cfg.data.batch_size * cfg.data.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(cfg.checkpointing.out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.model.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', cfg.data.dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - cfg.data.block_size, (cfg.data.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+cfg.data.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.data.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    if vocab_size is not None:
        cfg.model.vocab_size = vocab_size
        print(f"found vocab_size = {vocab_size} (inside {meta_path})")
    else:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")


start_iter: int = 0
if not cfg.model.checkpoint:
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = GPT(cfg)
else:
    print(f"Resuming training from {cfg.model.checkpoint}")
    # resume training from a checkpoint.
    checkpoint = torch.load(cfg.model.checkpoint, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['config']['model']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    cfg.model.layer = checkpoint_model_args["layer"]
    cfg.model.heads = checkpoint_model_args["heads"]
    cfg.model.embedding_size = checkpoint_model_args["embedding_size"]
    cfg.data.block_size = checkpoint["config"]["data"]["block_size"]
    cfg.model.bias = checkpoint_model_args["bias"]
    cfg.model.vocab_size = checkpoint_model_args["vocab_size"]

    # create the model
    model = GPT(cfg)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num: int = checkpoint['iter_num']
    start_iter = iter_num
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(cfg.model.dtype == 'float16'))

# optimizer
if cfg.optimizer.name == "adamw":
    optimizer = model.get_adamw_optimizer(cfg, device_type)
elif cfg.optimizer.name == "sgd":
    optimizer = model.get_sgd_optimizer(cfg, device_type)
else:
    raise ValueError(f"fInvalid optimizer name {cfg.optimizer.name}")

if cfg.model.checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if cfg.model.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval.iters)
        for k in range(cfg.eval.iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.lr_scheduler.warmup_iters:
        return cfg.optimizer.learning_rate * (it + 1) / (cfg.lr_scheduler.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_scheduler.decay_iters:
        return cfg.lr_scheduler.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.lr_scheduler.warmup_iters) / (cfg.lr_scheduler.decay_iters - cfg.lr_scheduler.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg.lr_scheduler.min_lr + coeff * (cfg.optimizer.learning_rate - cfg.lr_scheduler.min_lr)

# logging
if cfg.logging.wandb and master_process:
    import wandb
    wandb.init(project=cfg.wandb.project_name, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True))
    wandb.define_metric(name="iterations", step_metric="iter")

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
active_heads: int = cfg.model.heads

# import logging
# logging.basicConfig(level=logging.DEBUG)
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if cfg.lr_scheduler.decay else cfg.optimizer.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # apply head activation schedule
    if cfg.head_activation_schedule.get(iter_num, active_heads) != active_heads:
        new_active_heads = cfg.head_activation_schedule[iter_num]
        print(f"Changing active head configuration: {active_heads} -> {new_active_heads}")

        for decoder_block in raw_model.get_decoder_blocks():
            decoder_block.attn.set_active_heads(new_active_heads)
        active_heads = new_active_heads

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % cfg.eval.interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if cfg.logging.wandb:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "n_active_heads": active_heads,
            })

    if iter_num % cfg.checkpointing.interval == 0 and iter_num > start_iter and master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': OmegaConf.to_container(cfg, resolve=True),
        }
        print(f"saving checkpoint to {cfg.checkpoint_folder}")
        torch.save(checkpoint, cfg.checkpoint_folder / f"ckpt_{iter_num}iter.pt")
            
    if iter_num == 0 and cfg.eval.only:
            break
    
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(cfg.data.gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == cfg.data.gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / cfg.data.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if cfg.optimizer.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)

    # log gradients of all heads
    norms = {"iter": iter_num}
    for layer, decoder_block in enumerate(raw_model.get_decoder_blocks()):
        attn = decoder_block.attn

        if attn._c_attn.weight.grad is None:
            continue

        attention_head_norms = attn.get_attention_head_gradient_norms()

        for i in range(attn._total_heads):
            norms[f"gradient_norm/layer{layer:02}/c_attn/head{i:02}"] = attention_head_norms[i].item()

        layer_norm = torch.sum(attention_head_norms ** 2) ** (1. / 2)
        norms[f"gradient_norm/layer{layer:02}/c_attn/total"] = layer_norm

        if attn._c_proj.weight.grad is None:
            continue

        projection_head_norms = attn.get_projection_head_gradient_norms()

        for i in range(attn._total_heads):
            norms[f"gradient_norm/layer{layer:02}/c_proj/head{i:02}"] = projection_head_norms[i].item()

        layer_norm = torch.sum(projection_head_norms ** 2) ** (1. / 2)
        norms[f"gradient_norm/layer{layer:02}/c_proj/total"] = layer_norm

    if cfg.logging.wandb:
        wandb.log(norms)

    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.logging.interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * cfg.data.gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(cfg.data.batch_size * cfg.data.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > cfg.optimizer.max_iters:
        break

if ddp:
    destroy_process_group()
