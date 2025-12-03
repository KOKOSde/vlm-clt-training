#!/usr/bin/env python3
"""
Train a cross-layer sparse transcoder (CLT) for one LLaVA layer L.

Architecture (CLT-style, similar to Qwen CLT):
- Encoder:  LayerNorm(hidden_dim) → Linear(hidden_dim → feature_dim) → ReLU / TopK
- Decoder:  Linear(feature_dim → hidden_dim * n_targets)
            reshaped to [B, T, n_targets, hidden_dim] predicting:
                [MLP_L+1, MLP_L+2, ..., MLP_L+n_targets]

Data layout (see ACTIVATIONS.md):
- activations_dir/L{L}/batch_*_x.pt:  residual stream at layer L
- activations_dir/L{L}/batch_*_y.pt:  dict with:
      {
        'y': tensor([...]),        # primary target (L+1)
        'targets': [               # all CLT targets
            tensor([...]),         # MLP at L+1
            tensor([...]),         # MLP at L+2
            ...
        ],
        'n_targets': int,
      }
"""

import argparse
import glob
import json
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import threading
import queue


class Transcoder(nn.Module):
    """
    Cross-Layer Transcoder for one LLaVA layer L (multi-target CLT).

    - Reads residual stream at layer L
    - Writes to MLP outputs of multiple future layers (L+1..L+n_targets)
    """

    def __init__(
        self,
        hidden_dim: int,
        feature_dim: int,
        n_targets: int = 1,
        use_topk: bool = False,
        topk_pct: float = 0.12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.n_targets = max(1, int(n_targets))
        self.use_topk = use_topk
        self.topk_pct = topk_pct

        # Encoder: residual stream -> sparse features
        self.enc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Decoder: features -> residual stream for multiple future layers
        # Output shape before reshape: [B, T, hidden_dim * n_targets]
        self.dec = nn.Linear(feature_dim, hidden_dim * self.n_targets)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, H] residual stream at base layer L

        Returns:
            y_hat: [B, T, n_targets, H] predicted MLP outputs for L+1..L+n_targets
            z:     [B, T, feature_dim] sparse feature activations
        """
        # x: [B, T, H]
        z_pre = self.enc(x)  # [B, T, feature_dim] - pre-activation

        if self.use_topk:
            # TopK activation: keep only top K% of features
            k = max(1, int(z_pre.shape[-1] * self.topk_pct))
            # Get top-k values and indices
            topk_vals, topk_idx = torch.topk(z_pre, k, dim=-1)
            # Create sparse activation: zero everywhere except top-k
            z = torch.zeros_like(z_pre)
            z.scatter_(-1, topk_idx, torch.relu(topk_vals))
        else:
            # Standard ReLU activation
            z = torch.relu(z_pre)

        # Decode to multiple future layers
        y_hat_flat = self.dec(z)  # [B, T, H * n_targets]
        B, T, _ = y_hat_flat.shape
        y_hat = y_hat_flat.view(B, T, self.n_targets, self.hidden_dim)
        return y_hat, z


def load_batch_files(dir_L: str) -> List[str]:
    """Load all batch file paths for a layer."""
    x_files = sorted(glob.glob(os.path.join(dir_L, 'batch_*_x.pt')))
    y_files = sorted(glob.glob(os.path.join(dir_L, 'batch_*_y.pt')))
    
    # Extract start indices and match x/y pairs
    x_dict = {}
    for x_path in x_files:
        basename = os.path.basename(x_path)
        start_idx = basename.split('_')[1]
        x_dict[start_idx] = x_path
    
    y_dict = {}
    for y_path in y_files:
        basename = os.path.basename(y_path)
        start_idx = basename.split('_')[1]
        y_dict[start_idx] = y_path
    
    # Match pairs
    common_idx = sorted(set(x_dict.keys()) & set(y_dict.keys()))
    pairs = [(x_dict[idx], y_dict[idx]) for idx in common_idx]
    
    return pairs


def ensure_3d(t: torch.Tensor) -> torch.Tensor:
    """Handle tensor shapes: [N,1,T,H] -> [N,T,H] or [N,T,H] -> [N,T,H]"""
    if t.ndim == 4 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def infer_hidden_dim(sample_path: str) -> int:
    """Infer hidden dimension from a batch file."""
    obj = torch.load(sample_path, map_location='cpu')
    x = obj.get('x') if 'x' in obj else obj.get('data')
    x = ensure_3d(x)
    return int(x.shape[-1])


def load_and_unpack_batch(
    x_path: str,
    y_path: str,
    n_targets: int,
    cache: dict | None = None,
) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Load a batch file pair and unpack into individual samples.

    Returns a list of (x_sample, [y_sample_L+1, ..., y_sample_L+n_targets]) tuples.
    """
    # Check cache first
    cache_key = (x_path, y_path)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    
    try:
        x_obj = torch.load(x_path, map_location='cpu')
        y_obj = torch.load(y_path, map_location='cpu')
    except Exception as e:
        return []
    
    x_batch = x_obj.get('x') if 'x' in x_obj else x_obj.get('data')
    x_batch = ensure_3d(x_batch)  # [N, T, H]

    # Multi-target CLT format: 'targets' list + 'n_targets'
    targets_list = None
    if isinstance(y_obj, dict) and 'targets' in y_obj:
        targets_list = y_obj['targets']
    else:
        # Fallback: single-target format
        y_single = y_obj.get('y') if 'y' in y_obj else y_obj.get('data')
        y_single = ensure_3d(y_single)
        targets_list = [y_single]

    # Ensure 3D for each target and respect requested n_targets
    targets_list_3d: List[torch.Tensor] = [ensure_3d(t) for t in targets_list]
    if n_targets <= len(targets_list_3d):
        targets_list_3d = targets_list_3d[:n_targets]
    else:
        # If fewer targets than requested, pad with zeros like capture_activations.py
        base = targets_list_3d[0]
        for _ in range(n_targets - len(targets_list_3d)):
            targets_list_3d.append(torch.zeros_like(base))

    # Split into individual samples
    samples: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
    N = x_batch.shape[0]
    for i in range(N):
        x_sample = x_batch[i:i+1]  # [1, T, H]
        y_samples = [t[i:i+1] for t in targets_list_3d]  # list of [1, T, H]
        samples.append((x_sample, y_samples))
    
    # Cache if enabled
    if cache is not None:
        cache[cache_key] = samples
    
    return samples


def collate_samples(
    samples: List[Tuple[torch.Tensor, List[torch.Tensor]]],
    device: torch.device,
    target_len: int = 512,
    max_tokens: int = 4096,
    n_targets: int = 1,
):
    """
    Collate CLT samples into a batch, cropping to common length.

    Args:
        samples: list of (x, [y_1, ..., y_n_targets]) where each tensor is [1, T, H]
        device: torch device
        target_len: max sequence length to keep
        max_tokens: max tokens across batch (for safety)
        n_targets: number of CLT targets to collate

    Returns:
        X: [B, T_common, H]
        Ys: list of length n_targets, each [B, T_common, H]
    """
    xs = []
    ys_per_target: List[list] = [[] for _ in range(n_targets)]
    lengths = []
    
    for x, ys in samples:
        T = x.shape[1]
        lengths.append(T)
    
    if not lengths:
        raise RuntimeError("Empty batch")
    
    # Determine common take length
    common = min(min(lengths), target_len, max_tokens)
    
    for x, ys in samples:
        T = x.shape[1]
        if T == common:
            start = 0
        else:
            start = 0 if T < common else random.randint(0, T - common)
        xs.append(x[:, start:start+common, :])
        # Crop all targets with the same span
        for t_idx in range(n_targets):
            if t_idx < len(ys):
                y_t = ys[t_idx]
            else:
                # Safety: pad missing targets with zeros like capture_activations.py
                y_t = torch.zeros_like(ys[0])
            ys_per_target[t_idx].append(y_t[:, start:start+common, :])
    
    # Concatenate on CPU first, pin for fast H2D, then async transfer
    X_cpu = torch.cat(xs, dim=0).contiguous()
    Y_cpu_list = [torch.cat(buf, dim=0).contiguous() for buf in ys_per_target]
    
    if device.type == 'cuda':
        try:
            X_cpu = X_cpu.pin_memory()
            Y_cpu_list = [Y_cpu.pin_memory() for Y_cpu in Y_cpu_list]
        except Exception:
            pass
        X = X_cpu.to(device, non_blocking=True)
        Ys = [Y_cpu.to(device, non_blocking=True) for Y_cpu in Y_cpu_list]
    else:
        X = X_cpu
        Ys = Y_cpu_list
    
    return X, Ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--feature-dim', type=int, default=None)
    parser.add_argument('--cross-layer-depth', type=int, default=1, help='(LEGACY, unused) PLT: predict N layers ahead')
    parser.add_argument('--clt-mode', action='store_true', help='(LEGACY) kept for CLI compatibility; CLT mode is always used here')
    parser.add_argument('--use-topk', action='store_true', help='Use TopK activation to force exact sparsity')
    parser.add_argument('--topk-pct', type=float, default=0.12, help='TopK percentage (0.12 = 12%% sparsity)')
    parser.add_argument('--batch-samples', type=int, default=8, help='Number of samples per training batch')
    parser.add_argument('--reuse-steps', type=int, default=1, help='Reuse the same batch for N optimizer steps')
    parser.add_argument('--cache-batches', type=int, default=50, help='Cache up to N batch files in RAM')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16', 'auto'])
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--val-interval', type=int, default=200, help='Validation interval (0 to disable)')
    parser.add_argument('--val-samples', type=int, default=10, help='Number of validation samples')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    activations_dir = os.environ.get('ACTIVATIONS_DIR', cfg['outputs']['activations_dir'])
    transcoders_dir = cfg['outputs']['transcoders_dir']
    os.makedirs(transcoders_dir, exist_ok=True)
    
    # Optional logs directory
    logs_dir = cfg['outputs'].get('logs_dir', transcoders_dir)
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass

    L = args.layer
    dir_L = os.path.join(activations_dir, f'L{L}')
    batch_pairs = load_batch_files(dir_L)
    
    if not batch_pairs:
        raise RuntimeError(f"No batch files found at {dir_L}. Run capture script first.")
    
    print(f"Found {len(batch_pairs)} batch file pairs for layer {L}")

    # Split into train/val (90/10)
    random.shuffle(batch_pairs)
    split_idx = int(len(batch_pairs) * 0.9)
    train_batch_pairs = batch_pairs[:split_idx]
    val_batch_pairs = batch_pairs[split_idx:] if split_idx < len(batch_pairs) else batch_pairs[:min(2, len(batch_pairs))]

    hidden_dim = infer_hidden_dim(batch_pairs[0][0])
    feature_dim = args.feature_dim or int(cfg.get('feature_dim', 8192))

    # Infer number of CLT targets from first y batch (if present)
    first_y_obj = torch.load(batch_pairs[0][1], map_location='cpu')
    if isinstance(first_y_obj, dict) and 'n_targets' in first_y_obj and 'targets' in first_y_obj:
        n_targets = int(first_y_obj.get('n_targets', len(first_y_obj['targets'])))
    else:
        n_targets = 1
    
    print(f"Hidden dim: {hidden_dim}, Feature dim: {feature_dim}, n_targets: {n_targets}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Speed-friendly defaults on A100/H100
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass

    model = Transcoder(
        hidden_dim,
        feature_dim,
        n_targets=n_targets,
        use_topk=args.use_topk,
        topk_pct=args.topk_pct,
    ).to(device)
    
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass
    
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # Training settings
    target_len = int(cfg.get('seq_len', 512))

    # Select training dtype / AMP policy
    use_amp = (not args.no_amp) and device.type == 'cuda' and args.dtype in ('fp16', 'bf16')
    if args.dtype == 'fp16':
        amp_dtype = torch.float16
    elif args.dtype == 'bf16':
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    if device.type == 'cuda' and args.dtype in ('fp16', 'bf16'):
        model = model.to(amp_dtype)

    progress_path = os.path.join(logs_dir, f"progress_L{L}.txt")
    metrics_path = os.path.join(logs_dir, f"metrics_L{L}.json")

    pbar = tqdm(
        range(1, args.steps + 1),
        desc=f"L{L} training",
        unit="step",
        smoothing=0.1,
        dynamic_ncols=True,
        leave=True,
    )
    
    # Cache for batch files
    batch_cache = {} if args.cache_batches > 0 else None
    
    # Sample pool (unpacked samples from batches)
    all_train_samples = []
    all_val_samples = []
    
    cached_batch_data = None
    cached_until = 0

    # Track feature usage for dead feature detection
    feature_max_activations = torch.zeros(feature_dim, device=device)
    
    # Track metrics over time
    metrics_log = []
    
    # NEW: Track MLP neuron → CLT feature correlations for mapping
    # We'll compute correlation between MLP neurons and CLT features
    mlp_feature_correlations = torch.zeros((hidden_dim, feature_dim), device=device)
    mlp_feature_counts = torch.zeros((hidden_dim, feature_dim), device=device)

    # Prefetch and unpack batches asynchronously
    try:
        prefetch_threads_num = int(os.environ.get('PREFETCH_THREADS', '2'))
    except Exception:
        prefetch_threads_num = 2
    try:
        prefetch_queue_size = int(os.environ.get('PREFETCH_QUEUE_SIZE', '4'))
    except Exception:
        prefetch_queue_size = 4

    sample_pool_q: "queue.Queue[List[Tuple[torch.Tensor, List[torch.Tensor]]] | None]" = queue.Queue(maxsize=prefetch_queue_size)
    stop_prefetch = threading.Event()

    def prefetch_loop():
        try:
            while not stop_prefetch.is_set():
                # Pick a random batch file pair
                x_path, y_path = random.choice(train_batch_pairs)
                samples = load_and_unpack_batch(x_path, y_path, n_targets, cache=batch_cache)
                if samples:
                    sample_pool_q.put(samples)
        except Exception:
            pass

    prefetch_threads = [threading.Thread(target=prefetch_loop, daemon=True) for _ in range(prefetch_threads_num)]
    for t in prefetch_threads:
        t.start()

    # Pre-fill sample pool
    print("Pre-filling sample pool...")
    while len(all_train_samples) < args.batch_samples * 10:
        try:
            samples = sample_pool_q.get(timeout=5)
            all_train_samples.extend(samples)
        except Exception:
            # Fallback: load synchronously
            x_path, y_path = random.choice(train_batch_pairs)
            samples = load_and_unpack_batch(x_path, y_path, n_targets, cache=batch_cache)
            all_train_samples.extend(samples)
            break
    
    print(f"Sample pool ready: {len(all_train_samples)} samples")

    for step in pbar:
        # Refill sample pool from prefetch queue
        try:
            while not sample_pool_q.empty() and len(all_train_samples) < args.batch_samples * 20:
                samples = sample_pool_q.get_nowait()
                all_train_samples.extend(samples)
        except Exception:
            pass
        
        # If pool is low, load synchronously
        if len(all_train_samples) < args.batch_samples:
            x_path, y_path = random.choice(train_batch_pairs)
            samples = load_and_unpack_batch(x_path, y_path, n_targets, cache=batch_cache)
            all_train_samples.extend(samples)
        
        # Sample a batch
        if cached_batch_data is None or step > cached_until:
            batch_samples = random.sample(all_train_samples, k=min(args.batch_samples, len(all_train_samples)))
            X, Y_targets = collate_samples(
                batch_samples,
                device=device,
                target_len=target_len,
                max_tokens=4096,
                n_targets=n_targets,
            )
            if device.type == 'cuda' and args.dtype in ('fp16', 'bf16'):
                X = X.to(amp_dtype)
                Y_targets = [Y_t.to(amp_dtype) for Y_t in Y_targets]
            cached_batch_data = (X, Y_targets)
            cached_until = step + max(1, args.reuse_steps) - 1
        else:
            X, Y_targets = cached_batch_data

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                Y_hat, Z = model(X)  # Y_hat: [B, T, n_targets, H]

                # Reconstruction loss across all targets (like Qwen CLT)
                loss_rec = 0.0
                used_targets = 0
                for t_idx, Y_t in enumerate(Y_targets):
                    if t_idx < Y_hat.shape[2]:
                        y_pred = Y_hat[:, :, t_idx, :]
                        loss_rec = loss_rec + mse(y_pred, Y_t)
                        used_targets += 1
                if used_targets > 0:
                    loss_rec = loss_rec / used_targets
                else:
                    loss_rec = torch.tensor(0.0, device=X.device, dtype=X.dtype)

                loss_sparse = Z.abs().mean() * 1e-2
                loss = loss_rec + loss_sparse
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            Y_hat, Z = model(X)

            loss_rec = 0.0
            used_targets = 0
            for t_idx, Y_t in enumerate(Y_targets):
                if t_idx < Y_hat.shape[2]:
                    y_pred = Y_hat[:, :, t_idx, :]
                    loss_rec = loss_rec + mse(y_pred, Y_t)
                    used_targets += 1
            if used_targets > 0:
                loss_rec = loss_rec / used_targets
            else:
                loss_rec = torch.tensor(0.0, device=X.device, dtype=X.dtype)

            loss_sparse = Z.abs().mean() * 1e-2
            loss = loss_rec + loss_sparse
            loss.backward()
            opt.step()

        # Track feature activations
        with torch.no_grad():
            batch_max = Z.flatten(0, 1).max(dim=0).values
            feature_max_activations = torch.maximum(feature_max_activations, batch_max)
            
            # NEW: Track MLP→CLT correlations (VECTORIZED for speed)
            # For each timestep where a feature is active, record which MLP neurons are also active.
            # Use the primary target (L+1) as the MLP activation reference.
            if len(Y_targets) > 0:
                Y_main = Y_targets[0]  # [B, T, hidden_dim]
                Y_flat = Y_main.flatten(0, 1)  # [B*T, hidden_dim]
                Z_flat = Z.flatten(0, 1)       # [B*T, feature_dim]
                
                # Vectorized co-activation computation (match dtypes!)
                active_features = (Z_flat > 0).to(Y_flat.dtype)  # [B*T, feature_dim]
                # Compute weighted average: for each feature, average MLP activations when it's active
                # Y_flat.T @ active_features = [hidden_dim, feature_dim] - sum of MLP values when each feature is active
                mlp_feature_correlations += (Y_flat.abs().T @ active_features).to(mlp_feature_correlations.dtype)
                mlp_feature_counts += active_features.sum(dim=0).unsqueeze(0).to(mlp_feature_counts.dtype)
        
        # Compute L0 sparsity
        with torch.no_grad():
            l0_pct = (Z > 0).float().mean().item() * 100

        if step % 10 == 0 or step == args.steps:
            pbar.set_postfix({
                'rec': f"{loss_rec.item():.6f}",
                'sparse': f"{loss_sparse.item():.6f}",
                'L0%': f"{l0_pct:.1f}",
            })
            try:
                pct = (step / float(args.steps)) * 100.0
                with open(progress_path, 'w') as pf:
                    pf.write(f"{step}/{args.steps} ({pct:.1f}%) | L0: {l0_pct:.1f}% | Rec: {loss_rec.item():.6f}\n")
            except Exception:
                pass
        
        # Validation
        if args.val_interval > 0 and (step % args.val_interval == 0 or step == args.steps):
            model.eval()
            val_losses = []
            val_l0s = []
            
            # Load val samples if not loaded yet
            if not all_val_samples:
                for x_path, y_path in val_batch_pairs[:2]:
                    samples = load_and_unpack_batch(x_path, y_path, n_targets, cache=batch_cache)
                    all_val_samples.extend(samples)
            
            with torch.no_grad():
                for _ in range(min(args.val_samples, len(all_val_samples))):
                    try:
                        val_sample = [random.choice(all_val_samples)]
                        Xv, Yv_targets = collate_samples(
                            val_sample,
                            device=device,
                            target_len=target_len,
                            max_tokens=4096,
                            n_targets=n_targets,
                        )
                        if device.type == 'cuda' and args.dtype in ('fp16', 'bf16'):
                            Xv = Xv.to(amp_dtype)
                            Yv_targets = [Y_t.to(amp_dtype) for Y_t in Yv_targets]
                        Yv_hat, Zv = model(Xv)

                        # Validation loss over all targets
                        v_loss_rec = 0.0
                        v_used_targets = 0
                        for t_idx, Yv_t in enumerate(Yv_targets):
                            if t_idx < Yv_hat.shape[2]:
                                yv_pred = Yv_hat[:, :, t_idx, :]
                                v_loss_rec = v_loss_rec + mse(yv_pred, Yv_t).item()
                                v_used_targets += 1
                        if v_used_targets > 0:
                            v_loss_rec = v_loss_rec / v_used_targets

                        val_l0 = (Zv > 0).float().mean().item() * 100
                        val_losses.append(v_loss_rec)
                        val_l0s.append(val_l0)
                    except Exception:
                        continue
            
            model.train()
            
            if val_losses:
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_l0 = sum(val_l0s) / len(val_l0s)
                
                metrics_log.append({
                    'step': step,
                    'train_loss': loss_rec.item(),
                    'val_loss': avg_val_loss,
                    'train_l0': l0_pct,
                    'val_l0': avg_val_l0,
                })
                
                print(f"\n[Step {step}] Val Loss: {avg_val_loss:.6f} | Val L0: {avg_val_l0:.1f}%")

    # Stop prefetch
    try:
        stop_prefetch.set()
    except Exception:
        pass

    # Analyze dead features
    dead_threshold = 0.01
    dead_features = (feature_max_activations < dead_threshold).sum().item()
    dead_pct = (dead_features / feature_dim) * 100
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - Layer {L}")
    print(f"{'='*80}")
    print(f"Dead Features: {dead_features}/{feature_dim} ({dead_pct:.1f}%)")
    print(f"Final L0 Sparsity: {l0_pct:.1f}%")
    print(f"Final Reconstruction Loss: {loss_rec.item():.6f}")
    print(f"{'='*80}\n")

    # NEW: Compute final MLP→CLT mapping
    # Average correlations across all observations
    mlp_feature_mapping = torch.zeros_like(mlp_feature_correlations)
    for i in range(feature_dim):
        if mlp_feature_counts[:, i].sum() > 0:
            mlp_feature_mapping[:, i] = mlp_feature_correlations[:, i] / mlp_feature_counts[:, i].clamp(min=1)
    
    # Save checkpoint with mapping
    out_path = os.path.join(transcoders_dir, f"transcoder_L{L}.pt")
    torch.save({
        'layer': L,
        'hidden_dim': hidden_dim,
        'feature_dim': feature_dim,
        'state_dict': model.state_dict(),
        'training_metadata': {
            'steps': args.steps,
            'final_l0_pct': l0_pct,
            'dead_features': dead_features,
            'dead_pct': dead_pct,
            'final_rec_loss': loss_rec.item(),
        },
        # NEW: MLP→CLT mapping (which MLP neurons correlate with each CLT feature)
        'mlp_to_clt_mapping': mlp_feature_mapping.cpu(),  # [hidden_dim, feature_dim]
    }, out_path)
    print(f"✅ Saved transcoder with MLP→CLT mapping: {out_path}")
    print(f"   Mapping shape: {mlp_feature_mapping.shape} (MLP neurons × CLT features)")
    
    # Save mapping separately for easy access
    mapping_path = os.path.join(transcoders_dir, f"mapping_L{L}.pt")
    torch.save({
        'layer': L,
        'mlp_to_clt_mapping': mlp_feature_mapping.cpu(),  # [hidden_dim, feature_dim]
        'decoder_weights': model.dec.weight.detach().cpu(),  # [hidden_dim, feature_dim] - CLT→MLP
        'hidden_dim': hidden_dim,
        'feature_dim': feature_dim,
        'description': 'MLP neuron → CLT feature correlations from training data'
    }, mapping_path)
    print(f"✅ Saved mapping separately: {mapping_path}")
    
    # Save metrics
    if metrics_log:
        try:
            with open(metrics_path, 'w') as f:
                json.dump({
                    'layer': L,
                    'metrics': metrics_log,
                    'summary': {
                        'dead_features': dead_features,
                        'dead_pct': dead_pct,
                        'final_l0_pct': l0_pct,
                    }
                }, f, indent=2)
            print(f"✅ Saved metrics: {metrics_path}")
        except Exception as e:
            print(f"⚠️  Failed to save metrics: {e}")


if __name__ == '__main__':
    main()
















