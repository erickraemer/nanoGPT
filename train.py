import math
import sys
import time
from contextlib import nullcontext
from typing import Literal

import torch
from omegaconf import OmegaConf
from torch import Tensor
from wandb.plot import CustomChart

import wandb
from nanoGPT import util
from nanoGPT.causal_self_attention import CausalSelfAttention
from nanoGPT.data_loader import DataLoader
from nanoGPT.gpt import GPT
from nanoGPT.gpt_config import GPTConfig


class TrainEvalHandler:
    def __init__(self):
        self.model: GPT | None = None
        self.cfg: GPTConfig | None = None
        self.device: str | None = None
        self.data_loader: DataLoader | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler: torch.cuda.amp.GradScaler | None = None
        self.ctx: nullcontext | torch.amp.autocast | None = None
        self.iter_num: int = 0
        self.start_iter: int = 0
        self.best_val_loss: float = float('inf')
        self.last_norms: dict[str, int | float] = {}

    def init(self):

        # load config
        if len(sys.argv) < 2:
            raise RuntimeError("No config.yaml provided")

        arg = sys.argv[1]

        if '=' in arg:
            raise RuntimeError("Please provide the path to your config.yaml file as argument")
        cfg: GPTConfig = GPTConfig.load(arg)
        self.cfg = cfg

        if cfg.logging.wandb:
            wandb.init(project=cfg.wandb.project_name, name=cfg.wandb.run_name, config=OmegaConf.to_container(cfg, resolve=True))
            wandb.define_metric(name="iterations", step_metric="iter")

        # select device
        gpus = util.get_gpus()
        util.print_gpus(gpus)
        least_busy_gpu: util.NvidiaGPU = min(gpus, key=lambda gpu: gpu.rank())
        print(f"Using {least_busy_gpu.name} ({least_busy_gpu.id})\n")
        device = f'cuda:{least_busy_gpu.id}'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        torch.cuda.set_device(device)
        self.device = device

        checkpoint = None
        if cfg.model.checkpoint:
            print(f"Resuming training from {cfg.model.checkpoint}")
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
        else:
            print("Initializing a new model from scratch")

        model = GPT(cfg)
        model.to(device)
        self.model = model

        # set seed and tensor types
        torch.manual_seed(self.cfg.model.seed)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = 'cuda' if self.device.startswith('cuda') else 'cpu'  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.cfg.model.dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)#

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.amp.GradScaler(enabled=(self.cfg.model.dtype == 'float16'))
        self.scaler = scaler

        # optimizer
        if cfg.optimizer.name == "adamw":
            optimizer = model.get_adamw_optimizer(self.cfg, device_type)
        elif cfg.optimizer.name == "sgd":
            optimizer = model.get_sgd_optimizer(self.cfg, device_type)
        else:
            raise ValueError(f"fInvalid optimizer name {cfg.optimizer.name}")
        self.optimizer = optimizer

        if checkpoint is not None:
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            self.iter_num: int = checkpoint['iter_num']
            self.start_iter = self.iter_num
            self.best_val_loss = checkpoint['best_val_loss']

            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])

        # create data loader
        self.data_loader = DataLoader(self.cfg, device)

        if cfg.model.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(model)  # requires PyTorch 2.0

    @torch.no_grad()
    def _estimate_loss(self, dataset: Literal["train", "val"]) -> Tensor:

        model: GPT = self.model
        model.eval()

        losses = torch.zeros(self.cfg.eval.iters)
        for k in range(self.cfg.eval.iters):
            X, Y = self.data_loader.get_batch(dataset)
            with self.ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        loss = losses.mean()

        model.train()
        return loss

    def estimate_val_loss(self):
        return self._estimate_loss("val")

    def estimate_train_loss(self):
        return self._estimate_loss("train")

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.cfg.lr_scheduler.warmup_iters:
            return self.cfg.optimizer.learning_rate * (it + 1) / (self.cfg.lr_scheduler.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.cfg.lr_scheduler.decay_iters:
            return self.cfg.lr_scheduler.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.cfg.lr_scheduler.warmup_iters) / (
                    self.cfg.lr_scheduler.decay_iters - self.cfg.lr_scheduler.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.cfg.lr_scheduler.min_lr + coeff * (self.cfg.optimizer.learning_rate - self.cfg.lr_scheduler.min_lr)

    @torch.no_grad()
    def log_attention_head_norms(self, attn: CausalSelfAttention, layer: int) -> dict[str, float]:
        norms: dict[str, float] = {}

        attention_heads = attn.get_attention_head_gradients()
        attention_heads = attention_heads.reshape(len(attention_heads), -1)  # flatten

        c_attn_opt_state = self.optimizer.state[attn._c_attn.weight]
        v_sq = torch.sqrt(c_attn_opt_state["exp_avg_sq"] + self.optimizer.param_groups[0]['eps'])
        v_sq = attn.get_attention_head_view(v_sq)
        v_sq = v_sq.reshape(len(attention_heads), -1)  # flatten

        head_norms = torch.linalg.norm(attention_heads, dim=1)  # copy
        transformed_norms = torch.linalg.norm(attention_heads / v_sq, dim=1)  # copy

        for i in range(attn.total_heads):
            norms[f"gradient_norm/layer{layer:02}/c_attn/head{i:02}"] = head_norms[i].item()
            norms[f"transformed_norm/layer{layer:02}/c_attn/head{i:02}"] = transformed_norms[i].item()

        layer_norm = torch.linalg.norm(attention_heads)
        transformed_layer_norm = torch.linalg.norm(attention_heads / v_sq)
        norms[f"gradient_norm/layer{layer:02}/c_attn/total"] = layer_norm.item()
        norms[f"transformed_norm/layer{layer:02}/c_attn/total"] = transformed_layer_norm.item()

        return norms

    @torch.no_grad()
    def log_projection_head_norms(self, attn: CausalSelfAttention, layer: int) -> dict[str, float]:
        norms: dict[str, float] = {}

        projection_heads = attn.get_projection_head_gradients()
        projection_heads = projection_heads.reshape(len(projection_heads), -1) # flatten

        c_proj_opt_state = self.optimizer.state[attn._c_proj.weight]
        v_sq = torch.sqrt(c_proj_opt_state["exp_avg_sq"] + self.optimizer.param_groups[0]['eps'])
        v_sq = attn.get_projection_head_view(v_sq)
        v_sq = v_sq.reshape(len(projection_heads), -1)

        projection_head_norms = torch.linalg.norm(projection_heads, dim=1)  # copy
        transformed_norms = torch.linalg.norm(projection_heads / v_sq, dim=1)  # copy

        for i in range(attn.total_heads):
            norms[f"gradient_norm/layer{layer:02}/c_proj/head{i:02}"] = projection_head_norms[i].item()
            norms[f"transformed_norm/layer{layer:02}/c_proj/head{i:02}"] = transformed_norms[i].item()

        layer_norm = torch.linalg.norm(projection_heads)
        transformed_layer_norm = torch.linalg.norm(projection_heads / v_sq)
        norms[f"gradient_norm/layer{layer:02}/c_proj/total"] = layer_norm.item()
        norms[f"transformed_norm/layer{layer:02}/c_proj/total"] = transformed_layer_norm.item()

        return norms

    @torch.no_grad()
    def log_head_distributions(self, attn: CausalSelfAttention, layer: int) -> dict[str, float]:
        distributions: dict[str, float] = {}

        c_attn = attn._c_attn.weight.data.detach()
        heads = attn.get_attention_head_view(c_attn)
        heads = heads.reshape(len(heads), -1)  # flatten

        mean = torch.mean(heads, dim=1)
        variance = torch.std(heads, dim=1)

        for i in range(attn.total_heads):
            distributions[f"mean/layer{layer:02}/c_attn/head{i:02}"] = mean[i].item()
            distributions[f"variance/layer{layer:02}/c_attn/head{i:02}"] = variance[i].item()

        distributions[f"mean/layer{layer:02}/c_attn/total"] = torch.mean(c_attn, dim=(0, 1)).item()
        distributions[f"variance/layer{layer:02}/c_attn/total"] = torch.std(c_attn, dim=(0, 1)).item()

        return distributions

    @torch.no_grad()
    def log_metrics(self, model: GPT, iter_num: int):

        # log gradients of all heads
        metrics: dict[str, int | float | CustomChart] = {"iter": iter_num}
        for layer, block in enumerate(model.get_decoder_blocks()):
            attn: CausalSelfAttention = block.attn

            if self.cfg.metrics.attention_head_norms:
                attention_norms = self.log_attention_head_norms(attn, layer)
                metrics.update(attention_norms)

            if self.cfg.metrics.projection_head_norms:
                projection_norms = self.log_projection_head_norms(attn, layer)
                metrics.update(projection_norms)

            if self.cfg.metrics.attention_head_distribution:
                distributions = self.log_head_distributions(attn, layer)
                metrics.update(distributions)

        # calculate exponential moving average (ema)
        period: int = 1000
        alpha = 2 / (period + 1)

        for k, v in metrics.items():
            if k == "iter":
                continue

            if not isinstance(v, float) or isinstance(v, int):
                continue

            last_v = self.last_norms.get(k, v)
            metrics[k] = alpha * v + (1 - alpha) * last_v

        self.last_norms = metrics.copy()

        if len(metrics.keys()) <= 1:
            return

        # only return here to be able to debug this
        if not self.cfg.logging.wandb:
            return

        wandb.log(metrics)

    @torch.no_grad()
    def create_checkpoint(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
        }
        print(f"saving checkpoint to {self.cfg.checkpoint_folder}")

        iter = self.iter_num
        if iter % 1000 == 0:
            iter = f"{iter//1000}k"
        else:
            iter = str(iter)

        torch.save(checkpoint, self.cfg.checkpoint_folder / f"ckpt_{iter}.pt")

    @torch.no_grad()
    def eval(self, metadata: dict | None = None):

        if metadata is None:
            metadata = {}

        t_loss = self.estimate_train_loss()
        v_loss = self.estimate_val_loss()
        print(f"step {self.iter_num}: train loss {t_loss:.4f}, val loss {v_loss:.4f}")
        if not self.cfg.logging.wandb:
            return

        metrics = {
            "iter": self.iter_num,
            "loss/train": t_loss,
            "loss/val": v_loss,
            "n_active_heads": metadata.get("n_active_heads", self.cfg.model.heads)
        }

        metrics.update(metadata)
        wandb.log(metrics)

    @torch.no_grad()
    def head_dropout_eval(self):
        losses = {
            "iter": self.iter_num
        }

        baseline_loss = torch.zeros(self.cfg.eval.iters)
        dropout_loss = torch.zeros((self.cfg.model.layer, self.cfg.model.heads, self.cfg.eval.iters))
        for k in range(self.cfg.eval.iters):
            # use the same batch for all evals
            X, Y = self.data_loader.get_batch("val")

            with self.ctx:
                _, loss = self.model(X, Y)
            baseline_loss[k] = loss.item()

            for layer, decoder_block in enumerate(self.model.get_decoder_blocks()):
                # backup c_proj weight
                attn = decoder_block.attn
                weight = attn._c_proj.weight.data.clone()
                mask = attn._projection_head_mask.clone()

                for head in range(decoder_block.attn.total_heads):
                    # disable heads (this modifies the c_proj in this layer)
                    attn.set_disabled_heads([head])

                    with self.ctx:
                        _, loss = self.model(X, Y)
                    dropout_loss[layer, head, k] = loss.item()

                    # restore c_proj
                    attn._c_proj.weight.data = weight.clone()
                    attn._projection_head_mask.data = mask.clone()

        baseline_loss = baseline_loss.mean()
        dropout_loss = dropout_loss.mean(dim=2, keepdim=True)
        delta_loss = dropout_loss - baseline_loss

        for layer in range(self.cfg.model.layer):
            for head in range(self.cfg.model.heads):
                losses[f"head_importance/layer{layer:02}/head{head:02}"] = delta_loss[layer, head].item()

        if self.cfg.logging.wandb:
            wandb.log(losses)

    def train(self):
        X, Y = self.data_loader.get_batch('train')  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        running_mfu = -1.0
        active_heads: int = self.cfg.model.heads

        # references
        cfg = self.cfg
        optimizer = self.optimizer
        scaler = self.scaler
        model = self.model

        while True:

            # determine and set the learning rate for this iteration
            lr = self.get_lr(self.iter_num) if cfg.lr_scheduler.decay else cfg.optimizer.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # apply head activation schedule
            if cfg.head_activation_schedule.get(self.iter_num, active_heads) != active_heads:
                new_active_heads = cfg.head_activation_schedule[self.iter_num]
                print(f"Changing active head configuration: {active_heads} -> {new_active_heads}")

                for decoder_block in model.get_decoder_blocks():
                    decoder_block.attn.set_active_heads(range(new_active_heads))
                active_heads = new_active_heads

            do_eval: bool = self.iter_num % self.cfg.eval.interval == 0

            # evaluate the loss on train/val sets and write checkpoints
            if do_eval:
                self.eval({
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "n_active_heads": active_heads,
                })

                if self.cfg.metrics.head_dropout:
                    self.head_dropout_eval()

            if self.iter_num % cfg.checkpointing.interval == 0 and self.iter_num > self.start_iter:
                self.create_checkpoint()

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(cfg.data.gradient_accumulation_steps):
                with self.ctx:
                    logits, loss = model(X, Y)
                    loss = loss / cfg.data.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.data_loader.get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if cfg.optimizer.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)

            # log gradients to wandb
            if do_eval:
                self.log_metrics(model, self.iter_num)

            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % cfg.logging.interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * cfg.data.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = model.estimate_mfu(cfg.data.batch_size * cfg.data.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > cfg.optimizer.max_iters:
                break

def main():
    teh = TrainEvalHandler()
    teh.init()

    if teh.cfg.eval.only:
        teh.eval()
    else:
        teh.train()

if __name__ == "__main__":
    main()