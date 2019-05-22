# Installation

First the following requirements must be installed using Python 3.6 or higher:
```
torch>=1.0
scipy
pandas
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

# NMR data set
## Data
The preprocessed dataset (800MB) is here: [nmr.npy](https://www.dropbox.com/s/u0mw62dyfhcckg9/nmr.npy?dl=0).
Please save it into `Data/NMR/`.

The md5sum of the file is:
```
076a46bfd9a1e5362d597b634506685e  nmr.npy
```

## Training
```
cd Data/NMR
python learn.py --hidden_size 384  --num_layers 3 --head_radius 2 2 2 2 2 2 --folding nmr_folds_rep1.npy
```

# ChEMBL

## Data
The preprocessed dataset (8GB) is availabe here: [chembl80_data.npy](https://drive.google.com/file/d/1oPp1oAsgEwHc_-ESg5LE3B2ZGNV5Dl1r/view).
Please save it into `Data/chembl/`

The md5sum of the file is:
```
37cbd5b6767ebd0c869055353c61f1d9  chembl80_data.npy
```

## Training
```
cd Data/chembl
python learn.py --folding chembl80_folds_rep0.npy
```

