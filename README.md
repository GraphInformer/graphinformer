# Installation

First the following requirements must be installed using Python 3.6 or higher:
```
torch>=1.0
scipy
pandas
sklearn
tqdm
tensorboardX
```

The code also depends on rdkit, that is available from conda:
```
conda install rdkit -c rdkit
```

Then install the `graphinformer` package by
```
pip install -e .
```

# Training NMR data set
Preprocessed data and folds are included in the repository.
```
cd Data/NMR
python learn.py --hidden_size 384  --num_layers 3 --head_radius 2 2 2 2 2 2 --folding nmr_folds_rep1.npy
```

# Training ChEMBL data set
Preprocessed data and folds are included in the repository.

```
cd Data/chembl
python learn.py --folding chembl80_folds_rep0.npy
```

