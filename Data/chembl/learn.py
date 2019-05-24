from torch.utils.data import DataLoader
from rdkit import Chem
import graphinformer as gi
import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import sklearn.metrics
import chembl_loader

from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='ChEMBL prediction.')
parser.add_argument('-d', '--data_file', help='Data file', type=str, default="./chembl80_data.npy.bz2")
parser.add_argument('-f', '--folding_file', help='Folding file', type=str, default="./chembl80_folds_rep0.npy")
parser.add_argument('-r', '--head_radius', nargs='+', help='Head radius', type=int, default=[2, 2, 2, 2, 2, 2, 2, 2])
parser.add_argument('--init_norm', help='Initial weight for LayerNorm', type=float, default=0.01)
parser.add_argument('--init_norm_emb', help='Initial weight for Emb. LayerNorm', type=float, default=0.01)
parser.add_argument('-l', '--layers', help='Number of layers', type=int, default=3)
parser.add_argument('--hidden_size', help='Hidden size', type=int, default=192)
parser.add_argument('--attention_dropout', help='Attention dropout', type=float, default=0.1)
parser.add_argument('--final_dropout', help='Final dropout', type=float, default=0.1)
parser.add_argument('--hidden_dropout', help='Hidden dropout', type=float, default=0.1)
parser.add_argument('--embedding_dropout', help='Embedding dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', help='Weight decay', type=float, default=0.0)
parser.add_argument('-n', '--name', help="Name of run", type=str, default=None)
parser.add_argument('-b', '--batch_size', help="Batch size", type=int, default=32)
parser.add_argument('--learning_rate', help="Learning rate", type=float, default=1e-3)
parser.add_argument('--num_epochs', help="Number of epochs", type=int, default=100)
parser.add_argument('--residual', help="Use pure residual", type=int, default=1)
parser.add_argument('--pooler_post_layer', help="Add MLP after pooling", action="store_true")
parser.add_argument('--pooler_pre_layer', help="Add MLP before pooling", type=int, default=1)
parser.add_argument('--device', help="CUDA dev number", type=int, default=0)
parser.add_argument('--no_bar', help="Disables progress bar", action="store_true")
args = parser.parse_args()


if args.name is None:
    ## automatic title
    args.name = (
        f"gi{args.layers}"
        f"_{args.hidden_size}"
        f"_{''.join([str(i) for i in args.head_radius])}"
        f"_do{args.hidden_dropout}-{args.attention_dropout}-{args.final_dropout}"
        f"_res{int(args.residual)}"
        f"_wd{args.weight_decay}"
        f"_post{int(args.pooler_post_layer)}"
        f"_b{args.batch_size}"
    )
    if "chembl80" in args.data_file:
        args.name += "_c80"

args.fold = int(args.folding_file[-5])
args.run_folder = f"runs_gi_rep{args.fold}"

writer = SummaryWriter(f"{args.run_folder}/{args.name}")

print(args)
print("Run folder: " + args.run_folder)
print("Run name: " + args.name)

num_epochs = args.num_epochs
dev = f"cuda:{args.device}"

loader_tr, loader_va, loader_te = chembl_loader.loaders(
        data_file    = args.data_file,
        folding_file = args.folding_file,
        batch_size   = args.batch_size,
)

num_heads = len(args.head_radius)
key_size  = args.hidden_size // num_heads

config = gi.GIConfig(
    num_heads         = num_heads,
    key_size          = key_size,
    key_r_size        = key_size,
    value_size        = key_size,
    hidden_size       = args.hidden_size,
    intermediate_size = args.hidden_size * 2,
    route_size        = loader_tr.dataset.route_size,
    node_feature_size = loader_tr.dataset.node_feature_size,
    num_node_types    = max(loader_tr.dataset.node_types.values()) + 1,
    attention_dropout = args.attention_dropout,
    hidden_dropout    = args.hidden_dropout,
    embedding_dropout = args.embedding_dropout,
    final_dropout     = args.final_dropout,
    num_layers        = args.layers,
    initializer_range = 0.1,
    head_radius       = args.head_radius,
    init_norm         = args.init_norm,
    init_norm_emb     = args.init_norm_emb,
    use_route_values  = True,
    graph_num_labels  = loader_tr.dataset.mol_labels.shape[1],
    weight_decay      = args.weight_decay,
    learning_rate     = args.learning_rate,
    num_epochs        = args.num_epochs,
    residual          = args.residual > 0,
    pooler_post_layer = args.pooler_post_layer,
    pooler_pre_layer  = args.pooler_pre_layer >= 1,
    batch_size        = args.batch_size,
)
net  = gi.GIGraphClassification(config).to(dev)
loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

def roc_auc(y_true, y_score, ignore_index):
    keep = (y_true != ignore_index)
    return sklearn.metrics.roc_auc_score(
            y_true  = y_true[keep],
            y_score = y_score[keep]
    )

def evaluate(loader):
    ## evaluate data in loader
    net.eval()
    lsum   = 0.0
    lcount = 0
    targets_list = []
    inputs_list  = []
    with torch.no_grad():
        for b in loader:
            node_ids = b["node_ids"].to(dev)
            mask = (node_ids >= 1).float()
            pred = net(
                node_ids,
                node_features  = b["node_features"].to(dev),
                route_data     = b["route_features"].to(dev),
                attention_mask = mask,
                dists          = b["dists"].to(dev),
            )
            inputs = torch.stack([torch.zeros_like(pred), pred], dim=1)
            target = b["mol_labels"].to(dev).long()
            output = loss(inputs, target)

            targets_list.append(target)
            inputs_list.append(inputs)

            lsum   += output
            lcount += 1

    targets = torch.cat(targets_list, dim=0).cpu().numpy()
    inputs  = torch.cat(inputs_list, dim=0).cpu().numpy()
    aucs    = np.array([roc_auc(y_true  = targets[:,i], y_score = inputs[:,1,i], ignore_index=-100)
                 for i in range(inputs.shape[2])
              ])
    logloss = lsum.item() / lcount
    return {"logloss": logloss, "aucs": aucs}


def report_performance(epoch, live_logloss, best):
    res = {}
    res["tr"] = evaluate(loader_tr)
    res["va"] = evaluate(loader_va)
    res["te"] = evaluate(loader_te)
    for k in res.keys():
        writer.add_scalar(f"logloss/{k}", res[k]["logloss"], epoch)
        writer.add_scalar(f"auc_mean/{k}", res[k]["aucs"].mean(), epoch)
        writer.add_scalar(f"auc0/{k}", res[k]["aucs"][0], epoch)
        writer.add_scalar(f"auc1/{k}", res[k]["aucs"][1], epoch)
        writer.add_scalar(f"auc2/{k}", res[k]["aucs"][2], epoch)

    if len(best)==0 or (res["va"]["aucs"].mean() > best["va"]["aucs"].mean()):
        for k in res.keys():
            best[k] = res[k]
        best["epoch"] = epoch

    print(f"{epoch}. logloss=[{res['tr']['logloss']:.5f}, {res['va']['logloss']:.5f}, {res['te']['logloss']:.5f}] auc_mean=[{res['tr']['aucs'].mean():.5f}, {res['va']['aucs'].mean():.5f}, {res['te']['aucs'].mean():.5f}] (tr, va, te)")


optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[40, 70], gamma=0.3)

best = {}

for epoch in range(num_epochs):
    scheduler.step()
    net.train()

    loss_sum   = 0.0
    loss_count = 0

    for b in tqdm.tqdm(loader_tr, disable=args.no_bar):
        optimizer.zero_grad()
        node_ids = b["node_ids"].to(dev)
        mask = (node_ids >= 1).float()
        ## DEBUGGING: switching off partial charges
        #b["node_features"][:,:,20] = 0.0
        pred = net(
            node_ids,
            node_features  = b["node_features"].to(dev),
            route_data     = b["route_features"].to(dev),
            attention_mask = mask,
            dists          = b["dists"].to(dev),
        )
        inputs = torch.stack([torch.zeros_like(pred), pred], dim=1)
        target = b["mol_labels"].to(dev).long()
        output = loss(inputs, target)
        output.backward()

        optimizer.step()

        loss_sum   += output.detach()
        loss_count += 1

    loss_tr = loss_sum.item() / loss_count
    report_performance(epoch, loss_tr, best=best)

writer.close()
out_file = f"{args.run_folder}/{args.name}.run.npy"
np.save(out_file, {"config": config.__dict__, "best": best, "param_count": gi.count_parameters(net), "name": args.name})

print("Best result:")
print(f"{best['epoch']}. logloss=[{best['tr']['logloss']:.5f}, {best['va']['logloss']:.5f}, {best['te']['logloss']:.5f}] auc_mean=[{best['tr']['aucs'].mean():.5f}, {best['va']['aucs'].mean():.5f}, {best['te']['aucs'].mean():.5f}] (tr, va, te)")

print(f"Results saved into '{out_file}'.")
