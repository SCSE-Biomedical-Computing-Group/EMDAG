import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import re
from collections import defaultdict

# perf
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from emdag.datasets import get_dataset
from emdag.models import get_model
from emdag.utils.misc import *
from emdag.utils.data import *
from emdag.utils.train import *

# helpers
def _extract_any_stats(model):
    """
    Merge any dict-like debug stats exposed on model:
    - _last_std_stats
    - _last_extra_stats
    - _last_debug
    - _debug_stats
    Scans root and submodules (for DP/DDP/wrappers). Returns a flat dict of floats.
    """
    root = getattr(model, "module", model)
    keys = ["_last_std_stats", "_last_extra_stats", "_last_debug", "_debug_stats"]
    merged = {}

    def _maybe_merge(obj):
        for k in keys:
            d = getattr(obj, k, None)
            if isinstance(d, dict):
                for kk, vv in d.items():
                    try:
                        merged[str(kk)] = float(vv)
                    except Exception:
                        try:
                            merged[str(kk)] = float(getattr(vv, "item", lambda: vv)())
                        except Exception:
                            pass

    _maybe_merge(root)
    for sub in root.modules():
        _maybe_merge(sub)

    return merged


def _grad_norms_by_groups(model):
    """
    Compute grad L2 norms grouped by common module name fragments.
    Non-fatal if names don't exist. Returns dict of floats.
    """
    groups = {
        "seq_head": "g_seq_head",
        "pos_head": "g_pos_head",
        "rot_head": "g_rot_head",
        "encoder": "g_encoder",
        "trans_pos": "g_trans_pos",
        "trans_rot": "g_trans_rot",
        "trans_seq": "g_trans_seq",
    }
    matchers = {frag: re.compile(re.escape(frag)) for frag in groups.keys()}

    sums = defaultdict(float)
    for name, p in getattr(model, "named_parameters", lambda: [])():
        if p is None or p.grad is None:
            continue
        try:
            gnorm = float(p.grad.detach().norm(p=2).item())
        except Exception:
            continue
        for frag, rgx in matchers.items():
            if rgx.search(name):
                sums[groups[frag]] += gnorm

    return dict(sums)


def _gpu_mem_stats(device):
    if (isinstance(device, str) and device.startswith("cuda")) or (
        hasattr(device, "type") and device.type == "cuda"
    ):
        try:
            alloc = torch.cuda.memory_allocated()
            reserv = torch.cuda.memory_reserved()
            peak = torch.cuda.max_memory_allocated()
            return {
                "mem_alloc_mb": alloc / (1024**2),
                "mem_resvd_mb": reserv / (1024**2),
                "mem_peak_mb": peak / (1024**2),
            }
        except Exception:
            return {}
    return {}


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)

    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # grad accumulation steps (K)
    if args.grad_accum_steps is None:
        grad_accum_steps = int(getattr(config.train, "grad_accum_steps", 1))
    else:
        grad_accum_steps = int(args.grad_accum_steps)
    grad_accum_steps = max(1, grad_accum_steps)

    # Logging
    if args.debug:
        logger = get_logger("train", None)
        writer = BlackHole()
        log_dir = None
        ckpt_dir = None
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)

        ckpt_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        logger = get_logger("train", log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)

        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(
                args.config, os.path.join(log_dir, os.path.basename(args.config))
            )

    logger.info(args)
    logger.info(config)
    logger.info(f"grad_accum_steps = {grad_accum_steps}")

    # Data
    logger.info("Loading dataset...")
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)

    train_iterator = inf_iterator(
        DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            collate_fn=PaddingCollate(),
            shuffle=True,
            num_workers=args.num_workers,
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        collate_fn=PaddingCollate(),
        shuffle=False,
        num_workers=args.num_workers,
    )
    logger.info("Train %d | Val %d" % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info("Building model...")
    model = get_model(config.model).to(args.device)
    logger.info("Number of parameters: %d" % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad(set_to_none=True)
    it_first = 1

    # Resume
    if args.resume is not None or args.finetune is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        logger.info("Resuming from checkpoint: %s" % ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        it_first = ckpt["iteration"]
        model.load_state_dict(ckpt["model"])
        logger.info("Resuming optimizer states...")
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("Resuming scheduler states...")
        scheduler.load_state_dict(ckpt["scheduler"])

    # Train with grad accumulation
    def train(it):
        time_start = current_milli_time()
        model.train()

        K = grad_accum_steps
        optimizer.zero_grad(set_to_none=True)

        # Accumulate tensors for log_losses compatibility
        loss_sum = None                         # tensor scalar
        loss_dict_sum = {}                      # key -> tensor scalar
        last_batch_for_nan_dump = None
        time_forward_end = None

        for micro in range(K):
            batch = recursive_to(next(train_iterator), args.device)
            last_batch_for_nan_dump = batch

            loss_dict = model(batch)  # dict of tensor scalars
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)  # tensor scalar

            # scale for accumulation
            (loss / float(K)).backward()

            # accumulate detached tensors for logging
            loss_sum = loss.detach() if loss_sum is None else (loss_sum + loss.detach())

            for k, v in loss_dict.items():
                if not torch.is_tensor(v):
                    continue
                if k not in loss_dict_sum:
                    loss_dict_sum[k] = v.detach()
                else:
                    loss_dict_sum[k] = loss_dict_sum[k] + v.detach()

            if time_forward_end is None:
                time_forward_end = current_milli_time()

        if time_forward_end is None:
            time_forward_end = current_milli_time()

        # Grad norms before global clipping
        group_g = _grad_norms_by_groups(model)
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        time_backward_end = current_milli_time()

        # Gather extra stats
        extra_stats = _extract_any_stats(model)
        mem_stats = _gpu_mem_stats(args.device)

        # TB scalars
        for k, v in {**extra_stats, **group_g, **mem_stats}.items():
            try:
                writer.add_scalar(f"train/{k}", float(v), it)
            except Exception:
                pass

        # Compose averaged loss dict 
        loss_dict_avg = {}
        for k, v in loss_dict_sum.items():
            loss_dict_avg[k] = v / float(K)
        loss_dict_avg["overall"] = loss_sum / float(K)  

        others = {
            "grad": float(orig_grad_norm.detach().item()) if torch.is_tensor(orig_grad_norm) else float(orig_grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
            "time_forward": (time_forward_end - time_start) / 1000,
            "time_backward": (time_backward_end - time_forward_end) / 1000,
            "accum": K,
            "micro_bs": config.train.batch_size,
            "eff_bs": config.train.batch_size * K,
        }
        for d in (group_g, mem_stats, extra_stats):
            for k, v in d.items():
                try:
                    others[k] = float(v)
                except Exception:
                    pass

        log_losses(loss_dict_avg, it, "train", logger, writer, others=others)

        # NaN/Inf check 
        if not torch.isfinite(loss_dict_avg["overall"]):
            logger.error("NaN or Inf detected.")
            if not args.debug and log_dir is not None:
                torch.save(
                    {
                        "config": config,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "iteration": it,
                        "batch": recursive_to(last_batch_for_nan_dump, "cpu"),
                    },
                    os.path.join(log_dir, "checkpoint_nan_%d.pt" % it),
                )
            raise KeyboardInterrupt()

    # Validate
    def validate(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc="Validate", dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict["overall"] = loss
                loss_tape.update(loss_dict, 1)

                extra_stats = _extract_any_stats(model)
                for k, v in extra_stats.items():
                    try:
                        writer.add_scalar(f"val/{k}", float(v), it)
                    except Exception:
                        pass

        avg_loss = loss_tape.log(it, logger, writer, "val")
        if config.train.scheduler.type == "plateau":
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    # Loop
    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                if not args.debug and ckpt_dir is not None:
                    ckpt_path = os.path.join(ckpt_dir, "%d.pt" % it)
                    torch.save(
                        {
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "iteration": it,
                            "avg_val_loss": avg_val_loss,
                        },
                        ckpt_path,
                    )
    except KeyboardInterrupt:
        logger.info("Terminating...")
