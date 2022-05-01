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
1. prepare dataset for experiment 1 with 100 fishes and save it to `dataset.npy`
```{r}
 python prepare_data.py --f 100 --e 1
```  
2. load dataset `dataset.npy`, scale and split it into:
  * `fold_{1,5}.pkl`: 5 folders datasets for future work (hyperparameter tuning and model cross-validation
  * `scaled_train_test.pkl`: dataset for final training and testing (80-10) of model (once validated) for future work 
  * `scaled_train_val_test.pkl`: dataset for naive training-validation-testing (70-10-20) model prediction

```{r}
python data_loader.py --tr 80 --val 10 --fld 5
```

3. load `scaled_train_val_test.pkl`, consider traing and validation datasets for training, load model and save trained model to ''



