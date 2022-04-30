import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.animation as animation
import numpy as np

# get centroid points 
def traj_centre(trajectories):
    centroid_l = [trajectories[i].mean(axis=0) for i in range(trajectories.shape[0])]
    stack_centroid = np.vstack(centroid_l)
    return stack_centroid

# plot 3D trajectory of a fish
def plot_3D_traj(stack_centroid, title):
    x_centre, y_centre = stack_centroid[:,0], stack_centroid[:,1]
    time = [i/2 for i in range(len(stack_centroid))]
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_xlabel('x-coord')
    ax.set_ylabel('y-coord')
    ax.set_zlabel('time(frame)')
    ax.text(0., 1.05, 1.05, title, transform = ax.transAxes)
    ax.scatter(x_centre, y_centre, time, s=1)
    ax.scatter(x_centre, y_centre, s=1, c='red')
    return plt.show(block=False)


# plot density distribution fishes
def plot_hist2d(stack_traj_tot, size, title):
    x_tot = [stack_traj_tot[i, 0] for i in range(len(stack_traj_tot))]
    y_tot = [stack_traj_tot[i, 1] for i in range(len(stack_traj_tot))]   
    plt.hist2d(x_tot, y_tot, bins=(size, size), density=True, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title)
    return plt.show(block=False)


# plot experiment movie of zebrafish
def fish_movie_traj(trajectories, n_frames, interv, title):
    # First set up the frame coordinates
    frame = trajectories[1]
    centre = np.array(frame.mean(axis=0))
    x_frame, y_frame = frame[:, 0], frame[:, 1]

    # animation function.  This is called sequentially
    def animate(i):
        frame = trajectories[i]
        centre = np.array(frame.mean(axis=0))
        sc.set_offsets(frame)
        sc_centre.set_offsets(centre)
        tt.set_text(title + ' frame: ' + str(i)) 
        return sc, sc_centre, tt

    fig, ax = plt.subplots(figsize=(5,5)) 
    sc = ax.scatter(x_frame, y_frame, color ='green')
    sc_centre = ax.scatter(centre[0], centre[1], color ='red')
    tt = ax.text(0., 1.05, title, transform = ax.transAxes)
    # call the animator
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interv)
    return anim


def frame2traj(dict_frame, n_fish):
	fish_dict = {key:[] for key in range(n_fish)}
	frame_l = dict_frame.get('trajectories')
	for frame in frame_l:
		for i, row in enumerate(frame):
			fish_dict[i].append(row)
	return fish_dict


def conc_all_traj(fish_dict, n_fish):	
	# get probability
    traj_tot_l = [np.vstack(fish_dict[i]) for i in range(n_fish)]
    stack_traj_tot = np.vstack(traj_tot_l)
	# check for NaNs values
#    print('NaN Values:' + str(np.isnan(np.sum(stack_traj_tot))))
    stack = stack_traj_tot[~np.isnan(stack_traj_tot).any(axis=1), :]
    return stack



# plot experiment movie of zebrafish
def fish_movie_traj_vel(trajectories, n_frames, title, interv):
    # First set up the frame coordinates
    frame = np.array(trajectories.get(0))
    x, y = frame[:, 0], frame[:, 1]
    vx, vy = frame[:, 2], frame[:, 3]
    vtot = np.hstack((vx.reshape(-1,1), vy.reshape(-1,1)))
    colors = np.linalg.norm(vtot, axis=1)    

    # animation function.  This is called sequentially
    def animate(i):
        frame = np.array(trajectories.get(i))        
        x, y = frame[:, 0], frame[:, 1]
        vx, vy = frame[:, 2], frame[:, 3]
        vtot = np.hstack((vx.reshape(-1,1), vy.reshape(-1,1)))
        colors = np.linalg.norm(vtot, axis=1)

        tt.set_text(title + ' frame: ' + str(i)) 
        sc.set_offsets(frame)
        arr.set_offsets(frame)
        arr.set_UVC(vx, vy)
        arr.set_color(colormap(norm(colors)))
        return sc, arr
    
    colormap = cm.inferno
    norm = Normalize()
    norm.autoscale(colors)
    
    fig, ax = plt.subplots(figsize=(5,5)) 
    sc = ax.scatter(x, y, s=2, color ='green')
    arr = ax.quiver(x, y, vx, vy, color=colormap(norm(colors)))
    tt = ax.text(0., 1.05, 'Frame: ', transform = ax.transAxes)
    # call the animator
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interv)
    return anim


# get cosine similarity between fish velocities
def cosine_similarity(a, b):
    dot = np.dot(a, b.T)
    # vector norm    
    norma, normb = np.linalg.norm(a), np.linalg.norm(b)    
    cos = dot/(norma*normb)
    return cos


# plot experiment movie of zebrafish cosine
def fish_movie_parall_cos(trajectories, n_frames, title, interv):

    # First set up the frame coordinates
    frame = np.array(trajectories.get(0))
    x, y = frame[:, 0], frame[:, 1]
    vx, vy = frame[:, 2], frame[:, 3]   
    vxy = np.hstack([vx.reshape(-1,1), vy.reshape(-1,1)])

    # cosine similarity 
    score = cosine_similarity(vxy, vxy)
    score = np.where(score > 0, 1, score)
    score = np.where(score < 0, -1, score)
    n = score.shape[0]
    score[range(n), range(n)] = 0
    score_fish = np.mean(score, axis=1)

    # animation function.  This is called sequentially
    def animate(i):
        frame = np.array(trajectories.get(i))        
        x, y = frame[:, 0], frame[:, 1]
        vx, vy = frame[:, 2], frame[:, 3]
        vxy = np.hstack((vx.reshape(-1,1), vy.reshape(-1,1)))
        # cosine similarity
        score = cosine_similarity(vxy, vxy)
        score = np.where(score > 0, 1, score)
        score = np.where(score < 0, -1, score)
        n = score.shape[0]
        score[range(n), range(n)] = 0
        score_fish = np.mean(score, axis=1)
        cos_mean = np.mean(score)

        tt.set_text('Frame: ' + str(i) + ' cos_sim: ' + str(cos_mean))
        sc.set_offsets(frame)
        arr.set_offsets(frame)
        arr.set_UVC(vx, vy)
        arr.set_color(colormap(norm(score_fish)))
        return sc, arr
    
    colormap = cm.inferno
    norm = Normalize()
    norm.autoscale(score_fish)
    
    fig, ax = plt.subplots(figsize=(5,5)) 
    sc = ax.scatter(x, y, s=2, color ='green')
    arr = ax.quiver(x, y, vx, vy, color=colormap(norm(score_fish)))
    tt = ax.text(0., 1.05, 'Frame: ', transform = ax.transAxes)
    # call the animator
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interv)
    return anim

def plot_frame_parall_cos(trajectories, fr, pl):
    frame = np.array(trajectories.get(fr))
    x, y = frame[:, 0], frame[:, 1]
    vx, vy = frame[:, 2], frame[:, 3]   
    vxy = np.hstack([vx.reshape(-1,1), vy.reshape(-1,1)])
 
    score = cosine_similarity(vxy, vxy)
    score = np.where(score > 0, 1, score)
    score = np.where(score < 0, -1, score)
    n = score.shape[0]
    score[range(n), range(n)] = 0
    score_fish = np.mean(score, axis=1)

    sc = plt.scatter(x, y, s=2, color ='green')
    colormap = cm.inferno
    norm = Normalize()
    norm.autoscale(score_fish)
    
    if pl == 'quickver':
       plt.title('cosine similarity single frame')
       plt.quiver(x, y, vx, vy, color=colormap(norm(score_fish)))
       plt.colorbar()
    else:
       plt.title('heat matrix single frame')
       plt.imshow(score)
    return plt

