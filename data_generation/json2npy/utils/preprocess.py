import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import pickle
from .config_path import CONFIG_PATH
import yaml

def load_yaml_to_dicts(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    joint_names = {k: v for k, v in data.get('joint_names', {}).items()}
    color_map = {k: v for k, v in data.get('color_map', {}).items()}
    body_parts = {k: v for k, v in data.get('body_parts', {}).items()}

    return joint_names, color_map, body_parts

joint_names, color_map, body_parts = load_yaml_to_dicts(CONFIG_PATH)

def convert_json_2_npy(json_path, save_npy_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Check for missing keypoints
    missing_keypoints = []
    for frame_name, frame_data in data.items():
        for name in joint_names:
            if name not in frame_data:
                missing_keypoints.append((frame_name, name))

    if missing_keypoints:
        # raise ValueError(f"Missing keypoints detected.")
        return False

    # Create a keypoint color map
    # keypoint_color_map = {name: 'black' for name in joint_names}  # Default to black
    # for part, keys in body_parts.items():
    #     for key in keys:
    #         if key in joint_names:
    #             keypoint_color_map[key] = color_map[part]

    # Precompute offsets for each keypoint using numpy arrays
    num_frames = len(data)
    offsets = {name: np.zeros((num_frames, 3)) for name in joint_names}

    offsets_np = np.zeros((len(data.items()), len(joint_names), 3))

    for i, (frame_name, frame_data) in enumerate(data.items()):
        for j, name in enumerate(joint_names):
            if name in frame_data:
                offsets[name][i] = frame_data[name]
                offsets_np[i, j, :] = frame_data[name]

    # print(offsets_np.shape)

    np.save(save_npy_path, offsets_np)
    return True
