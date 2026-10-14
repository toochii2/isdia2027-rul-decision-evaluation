# Data

The study uses the NASA C-MAPSS turbofan engine degradation simulation data set (Saxena et al., 2008),
which is publicly available from NASA and is not redistributed here.

1. Download "CMAPSS Jet Engine Simulated Data" from the NASA open data portal:
   https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
2. Extract the twelve text files into `data/CMAPSSData/`:

```
data/CMAPSSData/
├── train_FD001.txt  test_FD001.txt  RUL_FD001.txt
├── train_FD002.txt  test_FD002.txt  RUL_FD002.txt
├── train_FD003.txt  test_FD003.txt  RUL_FD003.txt
└── train_FD004.txt  test_FD004.txt  RUL_FD004.txt
```

A different location can be passed to every target with `make ... DATA=/path/to/CMAPSSData`.
The raw data are needed only to retrain the models (`make models`); all analyses run from the
committed predictions in `results/predictions/`.

Reference: A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage propagation modeling for aircraft
engine run-to-failure simulation", in *Proc. International Conference on Prognostics and Health
Management*, Denver, CO, 2008.
