import pickle
import shutil
import json
from PIL import Image
import numpy as np
import torch


def save_image(img, filename):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()  # Move to CPU and convert to NumPy array
    # print( f"Image shape before reshape: {img.shape}")
    # Reshape the image to remove extra dimensions
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0,0]  # Remove the batch and channel dimensions
        # print("Image after dimension reduction:", img)

    # Convert to uint8 if necessary
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Save the image using Pillow
    image = Image.fromarray(img)
    image.save(filename)



def log_trajectory(trajectory, filename="trajectory_log.json"):
    """
    Logs a trajectory to a file in JSON format.

    Parameters:
    - trajectory: List of tuples containing (current_state, action, next_state, reward, done).
    - filename: Name of the file to log the trajectory.
    """
    # Convert the trajectory to a serializable format
    serializable_trajectory = [
        {
            "current_state": current_state,
            "action": action,
            "next_state": next_state,
            "reward": reward.tolist() if isinstance(reward, np.ndarray) else reward,
            "done": done
        }
        for current_state, action, next_state, reward, done in trajectory
    ]

    # Append the trajectory to the log file
    with open(filename, "a") as f:
        f.write(json.dumps(serializable_trajectory) + "\n")