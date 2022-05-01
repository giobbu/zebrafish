# COLLECTIVE MOTION IN ZEBRAFISHES
### Data
The data have been obtained using idtracker.ai. The data are related to trajectories for 60, 80 and 100 fish. Each subfolder has as a name the number of individuals in the trajectories.

* [Data can be downloaded here](https://drive.google.com/drive/folders/1Oq7JPmeY3bXqPXc_oTUwUZbHU-m4uq_5)

### Zebrafish data analysis:
`prob_simil`:
* Fishes probability density within tank
* Cosine similarity among fishes to assess how parallel are all fish to each other
<img src="./prob_simil/gif/cos_f100_e2.gif" width="400" height="400">

### ML model to predict fish trajectory and speed
`ml_model`:
* LSTM/GRU for multi-step ahead fish speed and trajectory prediction

##### Focal fish ID-0
<img src="./ml_model/predictions_4i_2o.gif" width="400" height="400">

##### Focal fish ID-20
<img src="./ml_model/predictions_4i_2o_f20.gif" width="400" height="400">

### References
This repository is based on the following works:
* [idTracker: tracking individuals in a group by automatic identification of unmarked animals](https://www.idtracker.es/) 
* [Deep attention networks reveal the rules of collective motion in zebrafish](https://journals.plos.org/ploscompbiol/article/authors?id=10.1371/journal.pcbi.1007354)
