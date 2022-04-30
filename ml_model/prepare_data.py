import numpy as np
import matplotlib.pyplot as plt
from utils import plot_frame_parall_cos, fish_movie_parall_cos, fish_movie_traj_vel, traj_centre, conc_all_traj, frame2traj, fish_movie_traj,  cosine_similarity
import matplotlib.animation as animation
import argparse
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--f', type=int, help='choose experiment based on number of fishes')
parser.add_argument('--e', type=int,  help='choose experiment 1 or 2 or 3')
args = parser.parse_args()

# load experiment zebrafish
frames_dict = np.load('folder_'+str(args.f)+'/'+ str(args.e)+'.npy', allow_pickle=True).item()

# get trajectories dict for each fish 
n_fish = frames_dict.get('trajectories').shape[1]
traj_dict = frame2traj(frames_dict, n_fish)

# get vtot=(vx, vy) by diff() for trajectories dict
traj_vel_dict = {key:[] for key in range(n_fish)}
n_frame = 2000000

# dict traj-vel for each fish
for fish in range(n_fish):
    traj = np.array(traj_dict.get(fish))
    # fill missing values back-forw filling
    traj = pd.DataFrame(traj).fillna(method="bfill", axis=0).fillna(method="ffill", axis=0).to_numpy()
    vel = np.diff(traj, axis=0)
    traj = traj[1:,:]
    traj_vel = np.hstack((traj, vel))
    traj_vel_dict[fish] = traj_vel
    if len(traj)<n_frame:
       n_frame = len(traj)  


fish_focal = traj_vel_dict[0]
traj_vel_dict.pop(0)
other_fishes = traj_vel_dict
dataset_l = []  

# get one fish array [x, y, vx, vy, rel_x, rel_y, rel_vx, rel_vy ...] 
for i in range(len(fish_focal)):
    # get features focal fish
    obs_foc = fish_focal[i,:].reshape(1,-1)
    x_foc, y_foc = obs_foc[:,0].reshape(-1,1), obs_foc[:,1].reshape(-1,1)
    vx_foc, vy_foc = obs_foc[:,2].reshape(-1,1), obs_foc[:,3].reshape(-1,1)

    # get other fishes features    
    rel_xy, rel_vxy = [],[]
    for fish in other_fishes.keys():

        # get other fish features
        other_fish = other_fishes[fish]
        obs_oth = other_fish[i,:].reshape(1,-1)
        # get coordinates and velocity
        x_oth, y_oth = obs_oth[:,0].reshape(-1,1), obs_oth[:,1].reshape(-1,1) 
        vx_oth, vy_oth = obs_oth[:,2].reshape(-1,1), obs_oth[:,3].reshape(-1,1)
        # get relative position and vel
        rel_x, rel_y = (x_foc - x_oth).reshape(-1,1), (y_foc - y_oth).reshape(-1,1)
        rel_vx, rel_vy = (vx_foc - vx_oth).reshape(-1, 1), (vy_foc - vy_oth).reshape(-1, 1)

        rel_xy.append(np.concatenate([rel_x, rel_y], axis=1))    
        rel_vxy.append(np.concatenate([rel_vx, rel_vy], axis=1))

    obs_rel_xy = np.hstack(rel_xy)
    obs_rel_vxy = np.hstack(rel_vxy)

    obs_foc_oth = np.hstack([x_foc, y_foc, vx_foc, vy_foc, obs_rel_xy, obs_rel_vxy])
    dataset_l.append(obs_foc_oth)

dataset = np.vstack(dataset_l)

print(dataset.shape)
print(obs_rel_xy.shape)
print(obs_rel_vxy.shape)

with open('dataset.npy', 'wb') as f:
    np.save(f, dataset)















