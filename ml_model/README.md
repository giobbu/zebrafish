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

* `train.py`: create model architecture from `models.py` and train it on the training-validation sets from `scaled_train_val_test.pkl`. Current model choices are LSTM and GRU encoder-decoder.
  * `--i`: input sequence (past observations)
  * `--o`: output sequence (future predictions)
  * `--b`: batch size
  * `--s`: set seed for reproducibility 
  * `--ut`: number of hidd units temporal layers
  * `--ufc`: number of hidd units fully connected
  * `--lr`: learning rate
  * `--dr`: drop rate
  * `--ep`: number of epochs
  * `--l1`: l1 regularizer
  * `--l2`: l2 regularizer
  * `--save`: name trained model to save

* `infer.py`: load trained model from `train.py` and evaluate it on test set from `scaled_train_val_test.pkl`. Visualize predictions (to be scaled back) both iteratively or with `matplotlib.animation`.
  * `--i`: input sequence
  * `--o`: output sequence
  * `--it`: iteratively visualize model predictions
  * `--n`: file name trained model

## Example

 
