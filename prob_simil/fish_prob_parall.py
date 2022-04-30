import numpy as np
import matplotlib.pyplot as plt
from utils import plot_frame_parall_cos, fish_movie_parall_cos, fish_movie_traj_vel, traj_centre, conc_all_traj, frame2traj, fish_movie_traj,  cosine_similarity
from utils import plot_3D_traj, plot_hist2d
import matplotlib.animation as animation
#import seaborn as sns
import argparse
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--f', type=int, default=60, help='choose experiment based on number of fishes')
parser.add_argument('--e', type=int, default=1, help='choose experiment 1 or 2 or 3')
parser.add_argument('--i', type=int, default=30, help='interval animation')
args = parser.parse_args()

# load experiment zebrafish
frames_dict = np.load('folder_'+str(args.f)+'/'+ str(args.e)+'.npy', allow_pickle=True).item()
 

########### Probability
# plot movie of experiment
interval = args.i
n_frames = frames_dict.get('trajectories').shape[0]
anim = fish_movie_traj(frames_dict.get('trajectories'), n_frames, interval, ' fishes trajectories')
plt.show()

# get centroid over time
stack_centroid = traj_centre(frames_dict.get('trajectories'))

# get 3D trajectory centroid
plot_3D = plot_3D_traj(stack_centroid, '3D centroid trajectory')
plt.show()

# get trajectories dict for each fish 
n_fish = frames_dict.get('trajectories').shape[1]
fish_dict = frame2traj(frames_dict, n_fish)

# stack all fish trajectories
stack_traj_tot = conc_all_traj(fish_dict, n_fish)

# the history of zebrafishes
hist1 = plot_hist2d(stack_traj_tot, 1000, 'fish universe')
plt.show()

hist2 = plot_hist2d(stack_traj_tot, 500, 'increase bin size')
plt.show()

hist3 = plot_hist2d(stack_traj_tot, 100, 'increase bin size')
plt.show()

hist4 = plot_hist2d(stack_traj_tot, 50, 'increase bin size')
plt.show()
hist5 = plot_hist2d(stack_traj_tot, 10, 'bin bang')
plt.show()

########## Cosine Similarity
# get trajectories dict for each fish 
n_fish = args.f
traj_dict = frame2traj(frames_dict, n_fish)

# get vtot=(vx, vy) by diff() for trajectories dict
traj_vel_dict = {key:[] for key in range(n_fish)}
n_frame = 2000000
for fish in range(n_fish):
    traj = np.array(traj_dict.get(fish))
    #traj = traj[~np.isnan(traj).any(axis=1), :] #traj[np.isnan(traj)] = 0 #traj[~np.isnan(traj).any(axis=1), :]    #traj = pd.DataFrame(traj).fillna(method="ffill").values
    traj = pd.DataFrame(traj).fillna(method="bfill", axis=0).fillna(method="ffill", axis=0).to_numpy()
    vel = np.diff(traj, axis=0)
    traj = traj[:-1, :]
    traj_vel = np.hstack([traj, vel])
    traj_vel_dict[fish] = np.vstack(traj_vel)
    if len(traj) < n_frame:
       n_frame = len(traj)  

# construct dict with positions and vel for all fishes each frame
frame_vel_dict = {key:[] for key in range(n_frame)}
for fish in range(n_fish):
    for frame in range(n_frame):
        frame_vel_dict[frame].append(traj_vel_dict.get(fish)[frame])    

# see animated frames trajectories colored by norm of vel
interv = args.i
anim = fish_movie_traj_vel(frame_vel_dict, n_frame,'fishes trajectories colored by vel norm', interv)
plt.show()

# see animated frame trajectories colored by cosine similarity
anim = fish_movie_parall_cos(frame_vel_dict, n_frame, 'fishes trajectories colored by cosine similarity', interv)
plt.show()

# check one frame quickver
fr = 1000
one_frame_quickver = plot_frame_parall_cos(frame_vel_dict, fr, 'quickver')
plt.show()

# check one frame matrix cosine_similarity
one_frame_matrix = plot_frame_parall_cos(frame_vel_dict, fr, 'heat')
plt.show()
