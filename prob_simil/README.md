# Data analysis

## Environment setup
create conda environment:
```{r}
conda env create -f environment.yml 
```

## Execute script
run command:
```{r}
python fish_prob_parall.py --f 100 --e 2 --i 20
```
notes:
* `--f`: select experiment based on number of fishes (i.e. 60, 80, or 100)
* `--e`: select experiment 1, 2, or 3
* `--i`: set interval between frames in animation




