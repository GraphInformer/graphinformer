import argparse
import random
import tqdm
from tensorboardX import SummaryWriter

import graphinformer as gi
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import nmr_loader

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--head_radius', nargs='+', help='Head radius', type=int, default=[2, 2, 2, 2, 2, 2])
parser.add_argument('--hidden_size', type=int, default=192, help='hidden size')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--folding', type=str, default="./nmr_folds_rep0.npy", help="folding npy file")
parser.add_argument('--no_bar', action='store_true', help='disable tqdm progress bar')
parser.add_argument('--hidden_dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--attention_dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--name', type=str, default=None,
                    help="Name of the run")
parser.add_argument('--normalize', action='store_true',
                    help="Add LayerNorm around attention")

dev = "cuda:0"

args = parser.parse_args()
print(args)

def evaluate(loader, model, criterion, n_atomtype):
    ## evaluate data in loader
    model.eval()
    lsum   = 0.0
    lcount = 0
    with torch.no_grad():
        for b in loader:
            n_graph = b["node_ids"].shape[0]
            n_node  = b["node_ids"].shape[1]
            y_onehot = torch.FloatTensor(n_graph, n_node, n_atomtype)
            y_onehot.zero_()
            y_onehot[np.repeat(range(n_graph),n_node), 
                               list(range(n_node))*n_graph, 
                               b["node_ids"].flatten()] = 1.0
            #mask = (b["node_ids"][:,1:] >= 1).float().to(dev)
            #node_type = y_onehot[:, 1:].to(dev)
            #features  = b["node_features"][:,1:].to(dev)
            mask = (b["node_ids"] >= 1).float().to(dev)
            node_type = y_onehot.to(dev)
            features  = b["node_features"].to(dev)
            target    = b["node_labels"].to(dev)
            dists     = b["dists"].to(dev)
            D         = b["route_features"].to(dev)

            inp  = torch.cat((node_type, features), -1)
            pred = model(inp, D, mask, dists).squeeze(-1)[:,1:]

            b_loss     = criterion(pred, target)
            loss_mask  = 1.0 - torch.isnan(target).float()
            losses     = b_loss[loss_mask == 1.0]

            lsum   += losses.sum().cpu().item()
            lcount += losses.shape[0]
    return lsum / lcount


def main(args):
    args.fold = int(args.folding[-5])
    args.run_folder = f"runs_rep{args.fold}"

    loader_tr, loader_va, loader_te = nmr_loader.loaders(
            data_file    = "./nmr.npy",
            folding_file = args.folding,
            batch_size   = args.batch_size,
    )

    n_atomtype = max(loader_tr.dataset.node_types.values()) + 1

    args.num_heads = len(args.head_radius)
    args.key_size  = args.hidden_size // args.num_heads
    args.key_r_size = args.key_size
    args.value_size = args.key_size
    args.input_size = loader_tr.dataset.node_feature_size + n_atomtype
    args.use_route_values = True
    args.route_size = loader_tr.dataset.route_size

    model = gi.GraphBlockNodeRegression(args).to(dev)
    #print(model)
    print(f"Parameter count: {gi.count_parameters(model)}")

    criterion = torch.nn.L1Loss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[40, 70], gamma=0.3)

    if args.name is None:
        if args.normalize:
            args.name = "gi_norm"
        else:
            args.name = "gi"
        args.name += f"_l{args.num_layers}_h{args.hidden_size}_d{args.hidden_dropout}_h{''.join([str(i) for i in args.head_radius])}"
    print(f"Run name is '{args.name}'.")
    writer = SummaryWriter(f"{args.run_folder}/{args.name}")

    def report_performance(epoch, live_mae_tr, best):
        mae_tr = evaluate(loader_tr, model, criterion, n_atomtype)
        mae_va = evaluate(loader_va, model, criterion, n_atomtype)
        mae_te = evaluate(loader_te, model, criterion, n_atomtype)
        writer.add_scalar(f"regression/mae_tr", mae_tr, epoch)
        writer.add_scalar(f"regression/mae_va", mae_va, epoch)
        writer.add_scalar(f"regression/mae_te", mae_te, epoch)
        if mae_va < best["mae_va"]:
            best["mae_tr"] = mae_tr
            best["mae_va"] = mae_va
            best["mae_te"] = mae_te
            best["epoch"]  = epoch

        print(f"{epoch}. mae_tr={mae_tr:.5f}, mae_va={mae_va:.5f}, mae_te={mae_te:.5f}")

    best = {"mae_va": np.inf}

    for epoch in range(0, args.epochs):
        model.train()
        scheduler.step()

        loss_sum   = 0.0
        loss_count = 0

        for i, b in enumerate(tqdm.tqdm(loader_tr, disable=args.no_bar)):
            optimizer.zero_grad()

            n_graph = b["node_ids"].shape[0]
            n_node  = b["node_ids"].shape[1]
            y_onehot = torch.FloatTensor(n_graph, n_node, n_atomtype)
            y_onehot.zero_()
            y_onehot[np.repeat(range(n_graph),n_node), 
                               list(range(n_node))*n_graph, 
                               b["node_ids"].flatten()] = 1.0
            #mask = (b["node_ids"][:,1:] >= 1).float().to(dev)
            #node_type = y_onehot[:, 1:].to(dev)
            #features  = b["node_features"][:,1:].to(dev)
            mask = (b["node_ids"] >= 1).float().to(dev)
            node_type = y_onehot.to(dev)
            features  = b["node_features"].to(dev)

            target    = b["node_labels"].to(dev)
            dists     = b["dists"].to(dev)
            D         = b["route_features"].to(dev)

            inp  = torch.cat((node_type, features), -1)
            pred = model(inp, D, mask, dists).squeeze(-1)[:,1:]

            b_loss     = criterion(pred, target)
            loss_mask  = 1.0 - torch.isnan(target).float()
            losses     = b_loss[loss_mask == 1.0]
            final_loss = losses.sum() / n_graph
            final_loss.backward()

            optimizer.step()

            loss_sum   += losses.detach().sum().cpu()
            loss_count += losses.shape[0]

        loss_tr = loss_sum.item() / loss_count
        report_performance(epoch, loss_tr, best)

    writer.close()
    np.save(f"{args.run_folder}/{args.name}.run.npy", {
        "config": args.__dict__,
        "best": best,
        "param_count": gi.count_parameters(model),
    })
    print(f"Best: Epoch {best['epoch']}. mae_tr={best['mae_tr']:.5f}, mae_va={best['mae_va']:.5f}, mae_te={best['mae_te']:.5f}")

if __name__ == "__main__":
    main(args)

