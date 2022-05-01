# ML model for fish speed and trajectories predictions

## Environment Installation
create and activate conda environment:
```{r}
conda env create -n tf-fish -f tf-environment.yml
conda activate tf-fish 
```



## ML experiments

* `prepare_data.py`: create dataset with coordinates and velocity of the focal fish  and the relative position and velocity of the other fishes.
  * `--f`: select experiment based on number of fishes (i.e. 60, 80, or 100)
  * `--e`: select experiment 1, 2, 3

* `data_loader.py`: scale and divide dataset in 1) naive temporal splitting, i.e., train-val-test, 2) train-test for assessing final performance, and 3) k folders for temporal cross-validation.
  * `--tr`: percentage training (default 80%)
  * `--val`: percentage validation (default 10%)
  * `--fld`: number of folders for temporal cross-validation (default 5)

* `train.py`:
  * `-- `
  *
  *

* `infer.py`:

