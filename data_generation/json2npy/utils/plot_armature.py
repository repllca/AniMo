import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


import pickle
# Define the keypoint names to display
from .config_path import CONFIG_PATH
from .preprocess import load_yaml_to_dicts

joint_names, color_map, body_parts = load_yaml_to_dicts(CONFIG_PATH)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_anim_npy(obj, use_json_path=True, save_as_gif=False, gif_filename='animation.gif'):
    if use_json_path:
        offsets_np = np.load(obj)
    else:
        offsets_np = obj
    assert isinstance(offsets_np, np.ndarray)
    
    print(offsets_np.shape)
    
    n_frame = offsets_np.shape[0]
    n_points = offsets_np.shape[1]
    n_dim = offsets_np.shape[2]
    assert n_dim == 3
    assert n_points >= 10
    assert n_frame > 1

    print("Data loaded successfully.")

    # Create a keypoint color map
    keypoint_color_map = {name: 'black' for name in joint_names}  # Default to black
    for part, keys in body_parts.items():
        for key in keys:
            if key in joint_names:
                keypoint_color_map[key] = color_map[part]

    # Initialize figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initialize scatter plot
    scatters = {name: ax.scatter([], [], [], color=color, label=name, alpha=1.0)
                for name, color in keypoint_color_map.items()}

    # Add a summarized legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = set()
    for label in labels:
        parts = label.split('def_')
        if len(parts) > 1:
            unique_labels.add(parts[1])
        else:
            unique_labels.add(parts[0])

    # Create legend with unique labels
    legend_handles = [handles[labels.index(next(filter(lambda x: unique_label in x, labels)))] for unique_label in unique_labels]
    if not save_as_gif:
        ax.legend(legend_handles, unique_labels, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Initialize text annotations
    annotations = [ax.text(0, 0, 0, str(i), fontsize=10, color='red') for i in range(n_points)]

    def update(frame):
        for i, (name, scatter) in enumerate(scatters.items()):
            x, y, z = offsets_np[frame, i, 0], offsets_np[frame, i, 1], offsets_np[frame, i, 2]
            scatter._offsets3d = (np.array([x]), np.array([y]), np.array([z]))
            annotations[i].set_position((x, y))
            annotations[i].set_3d_properties(z, 'z')
        return list(scatters.values()) + annotations

    # Skip frames to increase play speed
    skip_frames = 2
    ani = FuncAnimation(fig, update, frames=range(0, n_frame, skip_frames), interval=1, blit=False)

    if save_as_gif:
        ani.save(gif_filename, writer=PillowWriter(fps=15))
    else:
        ax.view_init(elev=20, azim=30)
        plt.show()

def test_show_plot():
    p = r'data_processed\aardvark_female__animationmotionextractedbehaviour.manisete15d87f2_aardvark_female_enrichmentboxshake_keypoints.json.npy'
    p = r'data_processed\african_buffalo_male__animationmotionextractedlocomotion.manisetfe33dea4_african_buffalo_male_walktoeat_keypoints.json.npy'
    
    plot_anim_npy(p)