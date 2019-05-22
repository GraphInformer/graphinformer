import graphinformer as gi
import numpy as np
from torch.utils.data import DataLoader

def loaders(data_file="./nmr.npy", folding_file="./folding.npy", batch_size=128, add_adj = False):
    nmr     = gi.MolDataset.from_npy(data_file)
    folding = np.load(folding_file, allow_pickle=True).item()
    
    data_tr = nmr.subset(folding["idx_train"])
    data_va = nmr.subset(folding["idx_valid"])
    data_te = nmr.subset(folding["idx_test"])

    loader_tr = DataLoader(
            dataset     = data_tr,
            batch_size  = batch_size,
            num_workers = 4,
            collate_fn  = lambda x: gi.mol_collate(x, add_adj = add_adj),
            shuffle     = True,
            pin_memory  = True,
    )
    loader_va = DataLoader(
            dataset     = data_va,
            batch_size  = batch_size,
            num_workers = 2,
            collate_fn  = lambda x: gi.mol_collate(x, add_adj = add_adj),
            shuffle     = False,
            pin_memory  = True,
    )
    loader_te = DataLoader(
            dataset     = data_te,
            batch_size  = batch_size,
            num_workers = 2,
            collate_fn  = lambda x: gi.mol_collate(x, add_adj = add_adj),
            shuffle     = False,
            pin_memory  = True,
    )
    return loader_tr, loader_va, loader_te


def loaders_ggnn(data_file="./nmr_webo.npy", folding_file="./folding.npy", batch_size=128,
                 all_route_features=False):
    nmr     = gi.MolDatasetGGNN.from_npy(data_file, all_route_features=all_route_features)
    folding = np.load(folding_file).item()
    
    data_tr = nmr.subset(folding["idx_train"])
    data_va = nmr.subset(folding["idx_valid"])
    data_te = nmr.subset(folding["idx_test"])

    loader_tr = DataLoader(
            dataset     = data_tr,
            batch_size  = batch_size,
            num_workers = 4,
            shuffle     = True,
            pin_memory  = True,
    )
    loader_va = DataLoader(
            dataset     = data_va,
            batch_size  = batch_size,
            num_workers = 2,
            shuffle     = False,
            pin_memory  = True,
    )
    loader_te = DataLoader(
            dataset     = data_te,
            batch_size  = batch_size,
            num_workers = 2,
            shuffle     = False,
            pin_memory  = True,
    )
    return loader_tr, loader_va, loader_te
