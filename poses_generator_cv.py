import os
import json
import cv2
import numpy as np
from PIL import Image

def read_sparse_file(sparse_file):
    """
    Read the sparse file to get the list of image indices.

    Parameters:
    - sparse_file: Path to the sparse file.

    Returns:
    - indices: List of image indices.
    """
    with open(sparse_file, 'r') as f:
        indices = [int(line.strip()) for line in f]
    return indices

def extract_camera_poses(scene_path, image_files):
    """
    Extract camera poses using ORB features and essential matrix.

    Parameters:
    - scene_path: Path to the scene directory containing images.
    - image_files: List of image file names.

    Returns:
    - poses: List of 4x4 numpy arrays representing camera poses.
    """
    images_path = os.path.join(scene_path, 'images')
    
    # Load the images
    images = []
    for img_file in image_files:
        img_path = os.path.join(images_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            continue
        images.append(img)
    
    if len(images) < 2:
        raise ValueError("Not enough images to compute camera poses.")
    
    # Detect ORB features
    orb = cv2.ORB_create()
    
    # Initialize storage for keypoints and descriptors
    keypoints_list = []
    descriptors_list = []
    
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
    
    # Initialize storage for camera poses
    poses = [np.eye(4)]  # First camera at origin
    
    # Match features between consecutive images and estimate poses
    for i in range(1, len(images)):
        # Match features between image i-1 and image i
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_list[i-1], descriptors_list[i])
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 5:
            print(f"Warning: Not enough matches between images {i-1} and {i}")
            continue
        
        # Extract points from the matches
        src_pts = np.float32([keypoints_list[i-1][m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_list[i][m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None or E.shape[0] == 0:
            print(f"Warning: Unable to compute essential matrix between images {i-1} and {i}")
            continue
        
        # Recover relative camera pose from the essential matrix
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)
        
        # Compose the transformation matrix (4x4)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        
        # Append the new pose
        poses.append(pose @ poses[-1])  # Chain the poses
    
    return poses

def load_image_data(scene_path, indices, poses):
    """
    Load image data and generate camera information.

    Parameters:
    - scene_path: Path to the scene directory containing images.
    - indices: List of image indices.
    - poses: List of 4x4 numpy arrays representing camera poses.

    Returns:
    - camera_data: List of dictionaries containing camera information.
    """
    images_path = os.path.join(scene_path, 'images')
    all_images = sorted(os.listdir(images_path))
    
    camera_data = []
    for i, idx in enumerate(indices):
        img_name = all_images[idx]
        img_path = os.path.join(images_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            continue
        height, width = img.shape[:2]

        # Extract position and rotation from the pose matrix
        position = poses[i][:3, 3].tolist()
        rotation = poses[i][:3, :3].tolist()

        # Generate random focal lengths for demonstration purposes
        fx = fy = 500 + np.random.rand() * 100

        camera_info = {
            "id": i,
            "img_name": img_name,
            "width": width,
            "height": height,
            "position": position,
            "rotation": rotation,
            "fy": fy,
            "fx": fx
        }
        camera_data.append(camera_info)
    return camera_data

def save_camera_data(camera_data, output_file):
    """
    Save camera data to a JSON file.

    Parameters:
    - camera_data: List of dictionaries containing camera information.
    - output_file: Path to the output JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(camera_data, f, indent=4)

if __name__ == "__main__":
    scene_path = 'data/truck_toy'
    sparse_num = 4
    sparse_file = os.path.join(scene_path, f'sparse_{sparse_num}.txt')
    output_file = os.path.join(scene_path, f'dust3r_{sparse_num}.json')

    indices = read_sparse_file(sparse_file)
    images_path = os.path.join(scene_path, 'images')
    all_images = sorted(os.listdir(images_path))
    image_files = [all_images[idx] for idx in indices]
    poses = extract_camera_poses(scene_path, image_files)
    camera_data = load_image_data(scene_path, indices, poses)
    save_camera_data(camera_data, output_file)

    print(f"Camera data saved to {output_file}")