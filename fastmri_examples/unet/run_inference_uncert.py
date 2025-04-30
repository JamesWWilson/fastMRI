#!/usr/bin/env python
# ==================================================================================
# run_inference_uncert_ALEX.py
# Purpose: Run MC-Dropout + evaluator on held-out fastMRI data
#          - save mean recon, MC stddev maps, and learned uncertainty maps
# ==================================================================================

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.models import Unet
from fastmri.models.uncertainty_evaluator import UncertaintyEvaluator

# -----------------------------------------------------------------------------


def enable_mc_dropout(module):
    """Put every Dropout layer into train() mode for MC sampling."""
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def run_evaluator(batch, model, evaluator, device, mc_runs):
    """
    Returns:
        mean_recon  (H,W)
        std_recon   (H,W)
        pred_unc    (H,W)
        slice_idx   int
        filename    str
    """
    image, _, mean, std, fname, slice_num, _ = batch

    # Monte-Carlo dropout samples
    samples = []
    for _ in range(mc_runs):
        out = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()
        samples.append(out)

    stack      = torch.stack(samples, 0)      # (R,1,H,W)
    mean_recon = stack.mean(0)                # (1,H,W)
    std_recon  = stack.std(0)                 # (1,H,W)

    # evaluator expects (B,1,H,W)
    inp = mean_recon.unsqueeze(0).to(device)
    pred_unc = evaluator(inp).squeeze().cpu().numpy()

    return (mean_recon.squeeze().numpy(),      # (H,W)
            std_recon.squeeze().numpy(),       # (H,W)
            pred_unc,                          # (H,W)
            int(slice_num[0]),
            fname[0])


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--challenge',
                        choices=('unet_knee_sc', 'unet_knee_mc', 'unet_brain_mc'),
                        default='unet_knee_sc')
    parser.add_argument('--gpus', type=int,
                        default=(torch.cuda.device_count() or 0),
                        help='How many GPUs to use (DataParallel); 0 = CPU')
    parser.add_argument('--state_dict_file', type=Path, default=None)
    parser.add_argument('--evaluator_ckpt',  type=Path, required=True)
    parser.add_argument('--data_path',       type=Path, required=True)
    parser.add_argument('--output_path',     type=Path, required=True)
    parser.add_argument('--mc_runs',         type=int, default=5)
    parser.add_argument('--batch_size',      type=int, default=1)
    parser.add_argument('--num_workers',     type=int, default=4)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Device + DataParallel
    if args.gpus > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'→ Using {device} (DataParallel on {args.gpus} GPU(s))')

    # -------------------------------------------------------------------------
    # 1) Load U-Net weights
    ckpt_u = torch.load(args.state_dict_file
                        or 'knee_sc_leaderboard_state_dict.pt',
                        map_location='cpu')
    sd_u = ckpt_u.get('state_dict', ckpt_u)
    sd_u = {k.replace('model.', '').replace('unet.', ''): v for k, v in sd_u.items()}

    net = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
    net.load_state_dict(sd_u, strict=False)
    enable_mc_dropout(net)

    # 2) Load evaluator weights
    ckpt_e = torch.load(args.evaluator_ckpt, map_location='cpu')
    sd_e = ckpt_e.get('state_dict', ckpt_e)
    sd_e = {k.replace('evaluator.', '', 1): v for k, v in sd_e.items()
            if k.startswith('evaluator.')}

    evaluator = UncertaintyEvaluator(chans=32, num_layers=4)
    evaluator.load_state_dict(sd_e, strict=False)

    # Move to device
    net.to(device).eval()
    evaluator.to(device).eval()

    # Wrap with DataParallel if >1 GPU
    if args.gpus > 1:
        devices = list(range(args.gpus))
        net = torch.nn.DataParallel(net, device_ids=devices)
        evaluator = torch.nn.DataParallel(evaluator, device_ids=devices)

    # -------------------------------------------------------------------------
    # 3) DataLoader
    mask = create_mask_for_mask_type('random', [0.08], [4])
    transform = T.UnetDataTransform('singlecoil', mask_func=mask)
    ds = SliceDataset(root=args.data_path, transform=transform, challenge='singlecoil')
    dl = torch.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers)

    # -------------------------------------------------------------------------
    # 4) Inference loop
    recons, mc_stds, uncs = defaultdict(list), defaultdict(list), defaultdict(list)

    for batch in tqdm(dl, desc='MC+Eval inference'):
        mean_rec, std_rec, unc_map, slc, fn = run_evaluator(
            batch, net, evaluator, device, args.mc_runs)
        recons[fn].append((slc, mean_rec))
        mc_stds[fn].append((slc, std_rec))
        uncs[fn].append((slc, unc_map))

    # -------------------------------------------------------------------------
    # 5) Save results
    out_root = Path(args.output_path)
    for coll, name in zip((recons, mc_stds, uncs),
                          ('reconstructions', 'mc_std', 'uncert_maps')):
        stacked = {f: np.stack([x for _, x in sorted(v)])
                   for f, v in coll.items()}
        fastmri.save_reconstructions(stacked, out_root / name)


if __name__ == '__main__':
    main()
