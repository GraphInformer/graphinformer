import graphinformer as gi
import numpy as np
from torch.utils.data import DataLoader

def loaders(data_file, folding_file, batch_size, num_workers=2, add_adj = False):
    mlr     = gi.MolDataset.from_npy(data_file)
    folding = np.load(folding_file, allow_pickle = True).item()
    
    data_tr = mlr.subset(folding["idx_tr"])
    data_va = mlr.subset(folding["idx_va"])
    data_te = mlr.subset(folding["idx_te"])

    loader_tr = DataLoader(
            dataset     = data_tr,
            batch_size  = batch_size,
            num_workers = num_workers,
            collate_fn  = lambda x: gi.mol_collate(x, add_adj = add_adj),
            shuffle     = True,
            pin_memory  = True,
    )
    loader_va = DataLoader(
            dataset     = data_va,
            batch_size  = batch_size,
            num_workers = num_workers,
            collate_fn  = lambda x: gi.mol_collate(x, add_adj = add_adj),
            shuffle     = False,
            pin_memory  = True,
    )
    loader_te = DataLoader(
            dataset     = data_te,
            batch_size  = batch_size,
            num_workers = num_workers,
            collate_fn  = lambda x: gi.mol_collate(x, add_adj = add_adj),
            shuffle     = False,
            pin_memory  = True,
    )
    return loader_tr, loader_va, loader_te

def loaders_ggnn(data_file, folding_file, batch_size, num_workers=2,
                 all_route_features=False):
    dataset = gi.MolDatasetGGNN.from_npy(data_file, all_route_features=all_route_features)
    folding = np.load(folding_file, allow_pickle = True).item()
    
    data_tr = dataset.subset(folding["idx_tr"])
    data_va = dataset.subset(folding["idx_va"])
    data_te = dataset.subset(folding["idx_te"])

    loader_tr = DataLoader(
            dataset     = data_tr,
            batch_size  = batch_size,
            num_workers = num_workers,
            shuffle     = True,
            pin_memory  = True,
    )
    loader_va = DataLoader(
            dataset     = data_va,
            batch_size  = batch_size,
            num_workers = num_workers,
            shuffle     = False,
            pin_memory  = True,
    )
    loader_te = DataLoader(
            dataset     = data_te,
            batch_size  = batch_size,
            num_workers = num_workers,
            shuffle     = False,
            pin_memory  = True,
    )
    return loader_tr, loader_va, loader_te
