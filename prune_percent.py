import os
import sys
import torch
from argparse import ArgumentParser
from utils.gaussian_model import GaussianModel
from utils.net_utils import PrunePredictor

def prune_pure_feature(args):
    torch.cuda.set_device(args.data_device)

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(args.ply_path)
    
    perdictor = PrunePredictor(input_dim=args.input_dim).cuda()
    perdictor.load_model(args.net_weights_path, args.data_device)
    perdictor.eval()
    
    with torch.no_grad():
        gaussians.get_prune_input_f15(args.knn_k, args.knn_method)
        scores = perdictor(gaussians.prune_features)[:, 0]

    N = gaussians.get_point_number()
    keep_num = int(N * args.keep_percent)
    
    sorted_indices = scores.argsort(descending=True)
    keep_indices = sorted_indices[:keep_num]
    valid_mask = torch.zeros(N, dtype=torch.bool, device=scores.device)
    valid_mask[keep_indices] = True
    gaussians.prune_points(valid_mask)
    
    gaussians.save_ply(args.output_ply_path)
    print(f"Saved {args.output_ply_path}")
    
    score_path = args.output_ply_path.replace(".ply", ".npy")
    import numpy as np
    np.save(score_path, scores.cpu().numpy())
    print(f"Saved scores to {score_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--ply_path", required=True, type=str, default="")
    parser.add_argument("--output_ply_path", required=True, default="", type=str)
    parser.add_argument('--keep_percent', required=True, default="0.8", type=float, help='keep percentage')
    parser.add_argument("--input_dim", type=int, default=15)
    parser.add_argument("--knn_k", type=int, default=128)
    parser.add_argument("--knn_method", type=str, default="ivf", choices=["ivf", "brute_force", "ckdtree"])
    parser.add_argument("--net_weights_path", type=str, default="")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--data_device", type=str, default="cuda:0")
    args = parser.parse_args(sys.argv[1:])
    
    if args.net_weights_path == "":
        script_dir = os.path.dirname(__file__)
        args.net_weights_path = os.path.join(script_dir, "net_weights", "net_f15.pth")
    print("Calculating " + args.ply_path)

    prune_pure_feature(args)
