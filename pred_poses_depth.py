import os
import numpy as np
import cv2
import json
import torch
import open3d as o3d
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import sys
sys.path.append('/ec/pdx/disks/il_ailabs/ldgomezr/Depth-Anything-V2')

from depth_anything_v2.dpt import DepthAnythingV2 # Ensure this is correctly imported based on your setup

from tqdm import tqdm
import itertools
import pdb

# Initialize DepthAnythingV2 with specified encoder and device
def initialize_depth_estimator(encoder='vitl', checkpoint_path='checkpoints/depth_anything_v2_vitl.pth'):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the depth model
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    return depth_anything, DEVICE

def select_images_for_uniform_distribution(image_folder, num_desired_images):
    # Get image files from folder
    image_files = sorted([x for x in os.listdir(image_folder) if x.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Initialize SIFT detector (more robust for feature detection)
    sift = cv2.SIFT_create()

    # Calculate feature descriptors for all images
    descriptors = []
    for img_file in tqdm(image_files, desc="Calculating SIFT descriptors"):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to load image at {img_path}")
            descriptors.append(None)
            continue

        # Detect and compute features
        _, des = sift.detectAndCompute(img, None)
        descriptors.append(des)

    # Calculate pairwise distances between all image descriptors using feature matching
    distances = np.zeros((len(image_files), len(image_files)))  # Distance matrix
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    for i in range(len(image_files)):
        for j in range(i + 1, len(image_files)):
            des_i, des_j = descriptors[i], descriptors[j]
            if des_i is not None and des_j is not None:
                matches = bf.match(des_i, des_j)
                distances[i, j] = distances[j, i] = -len(matches)  # Negative similarity for sorting

    # Select images by maximizing distribution based on distances
    step_size = max(1, len(image_files) // num_desired_images)
    selected_indices = [i * step_size for i in range(num_desired_images)]
    selected_images = [image_files[i] for i in selected_indices]

    return selected_images, selected_indices


def select_images_by_feature_difference(image_folder, num_desired_images):
    # Get image files from folder
    image_files = sorted([x for x in os.listdir(image_folder) if x.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Store descriptors and image indices
    descriptors = []
    similarities = np.zeros((len(image_files), len(image_files)))  # Similarity matrix

    # Step 1: Calculate feature descriptors for all images
    for idx, img_file in enumerate(tqdm(image_files, desc="Analyzing features")):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to load image at {img_path}")
            descriptors.append(None)
            continue

        # Detect and compute features
        kp, des = orb.detectAndCompute(img, None)
        descriptors.append(des)

    # Step 2: Compute similarity scores between all image pairs
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i, j in itertools.combinations(range(len(image_files)), 2):
        des_i, des_j = descriptors[i], descriptors[j]
        if des_i is not None and des_j is not None:
            matches = bf.match(des_i, des_j)
            similarities[i, j] = len(matches)
            similarities[j, i] = len(matches)  # Symmetric similarity

    # Step 3: Select images with best distribution
    selected_indices = [0]  # Start with the first image
    for _ in range(1, num_desired_images):
        last_selected = selected_indices[-1]
        
        # Find the image that has the lowest similarity to the most recently selected image
        distances = similarities[last_selected, :]
        # Set similarity of already selected images to a high value to avoid re-selecting them
        distances[selected_indices] = np.inf
        next_index = np.argmin(distances)
        
        selected_indices.append(next_index)

    # Map indices to filenames
    selected_images = [image_files[i] for i in selected_indices]

    return selected_images, selected_indices

def get_selected_images(scene_path, image_folder, num_desired_images):
    # Define el nombre del archivo en función del número de imágenes deseadas
    selected_images_file = os.path.join(scene_path, f'desired_images_{num_desired_images}.txt')
    
    # Verifica si el archivo ya existe
    if os.path.exists(selected_images_file):
        # Lee el archivo y carga los nombres de las imágenes
        with open(selected_images_file, 'r') as f:
            selected_image_files = f.read().splitlines()
        print(f"Loaded selected images from {selected_images_file}")
    else:
        # Genera los nombres de las imágenes seleccionadas
        selected_image_files, _ = select_images_for_uniform_distribution(image_folder, num_desired_images)
        
        # Guarda los nombres de las imágenes en el archivo
        with open(selected_images_file, 'w') as f:
            f.write("\n".join(selected_image_files))
        print(f"Saved selected images to {selected_images_file}")
    
    return selected_image_files


# Step 1: Load Images and Masks
def load_images_and_masks(image_folder, mask_folder):
    image_files = sorted(os.listdir(image_folder))
    images = [cv2.imread(os.path.join(image_folder, img_file)) for img_file in image_files]
    masks = [cv2.imread(os.path.join(mask_folder, os.path.splitext(img_file)[0] + '.png'), cv2.IMREAD_GRAYSCALE) for img_file in image_files]
    return images, masks, image_files

# Step 2: Generate Depth Maps
def generate_depth_maps(images, masks, depth_estimator, device, input_size=518, scene_path='.'):
    depth_maps = []
    for image, mask in zip(images, masks):
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to RGB if necessary
        masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        
        # Estimate depth
        with torch.no_grad():
            depth_map = depth_estimator.infer_image(masked_image_rgb, input_size=input_size)
        
        # Convert depth map to numpy array
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.squeeze().cpu().numpy()
        
        # Save depth map for visualization
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = depth_map_normalized.astype(np.uint8)
        cv2.imwrite(os.path.join(scene_path, 'depth_maps/depth_{}.png'.format(len(depth_maps))), depth_map_normalized)
        
        depth_maps.append(depth_map)
    
    return depth_maps


# Step 3: Generate Point Clouds from Depth Maps
def generate_point_clouds(images, masks, depth_maps, K):
    point_clouds = []
    for image, mask, depth_map in zip(images, masks, depth_maps):
        # Get valid depth pixels
        mask_indices = np.where(mask > 0)
        depths = depth_map[mask_indices[0], mask_indices[1]]
        u = mask_indices[1]
        v = mask_indices[0]
        # Back-project to 3D
        x = (u - K[0, 2]) * depths / K[0, 0]
        y = (v - K[1, 2]) * depths / K[1, 1]
        z = depths
        points = np.stack((x, y, z), axis=-1)
        # Get colors
        colors = image[mask_indices[0], mask_indices[1]] / 255.0
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        point_clouds.append(pcd)
    return point_clouds

def visualize_matches(img1, img2, kp1, kp2, matches, index, scene_path):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matches_path = os.path.join(scene_path, 'matches')

    cv2.imwrite(os.path.join(matches_path, 'matches_{}.png'.format(index)), img_matches)


def estimate_poses(images, depth_maps, K, scene_path):
    poses = [np.eye(4)]
    for i in range(1, len(images)):
        img1 = images[i - 1]
        img2 = images[i]
        depth1 = depth_maps[i - 1]
        depth2 = depth_maps[i]

        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect and compute features using SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # Match features using FLANN-based matcher
        if des1 is None or des2 is None:
            print(f"No descriptors found in images at index {i - 1} and {i}. Skipping.")
            poses.append(poses[-1])
            continue

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            print(f"Not enough good matches between images at index {i - 1} and {i}. Skipping.")
            poses.append(poses[-1])
            continue

        visualize_matches(img1, img2, kp1, kp2, good_matches, i, scene_path)

        # Get 2D keypoint coordinates
        pts1_2d = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2_2d = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Get depth values at keypoint locations
        depths1 = depth1[pts1_2d[:, 1].astype(int), pts1_2d[:, 0].astype(int)]
        depths2 = depth2[pts2_2d[:, 1].astype(int), pts2_2d[:, 0].astype(int)]

        # Filter out points with invalid depth
        valid_idx = (depths1 > 0) & (depths2 > 0)
        if np.sum(valid_idx) < 4:
            print(f"Not enough valid depth points between images at index {i - 1} and {i}. Skipping.")
            poses.append(poses[-1])
            continue

        pts1_2d = pts1_2d[valid_idx]
        pts2_2d = pts2_2d[valid_idx]
        depths1 = depths1[valid_idx]
        depths2 = depths2[valid_idx]

        # Back-project to 3D
        pts1_3d = cv2.undistortPoints(np.expand_dims(pts1_2d, axis=1), K, None).squeeze()
        pts1_3d = np.hstack((pts1_3d, np.ones((pts1_3d.shape[0], 1))))
        pts1_3d = pts1_3d * depths1[:, np.newaxis]

        pts2_3d = cv2.undistortPoints(np.expand_dims(pts2_2d, axis=1), K, None).squeeze()
        pts2_3d = np.hstack((pts2_3d, np.ones((pts2_3d.shape[0], 1))))
        pts2_3d = pts2_3d * depths2[:, np.newaxis]

        # Estimate transformation using RANSAC
        retval, out_transform, inliers = cv2.estimateAffine3D(pts1_3d[:, :3], pts2_3d[:, :3], ransacThreshold=0.01)

        if retval:
            transform = np.eye(4)
            transform[:3, :] = out_transform
            poses.append(transform @ poses[-1])
        else:
            print(f"Transformation estimation failed between images at index {i - 1} and {i}. Skipping.")
            poses.append(poses[-1])

    return poses


# Step 5: Invert Poses to Simulate Camera Movement
def invert_poses(poses):
    inverted_poses = []
    for pose in poses:
        R = pose[:3, :3]
        t = pose[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        pose_inv = np.eye(4)
        pose_inv[:3, :3] = R_inv
        pose_inv[:3, 3] = t_inv
        inverted_poses.append(pose_inv)
    return inverted_poses

# Step 6: Generate Camera Poses JSON
def generate_camera_poses_json(poses, image_files, images, K, output_path):
    cameras = []
    fx = K[0, 0]
    fy = K[1, 1]
    for i, (pose, img_file, image) in enumerate(zip(poses, image_files, images)):
        position = pose[:3, 3].tolist()
        rotation = pose[:3, :3].tolist()
        width = image.shape[1]
        height = image.shape[0]
        camera = {
            "id": i,
            "img_name": img_file,
            "width": width,
            "height": height,
            "position": position,
            "rotation": rotation,
            "fx": fx,
            "fy": fy
        }
        cameras.append(camera)
    with open(output_path, "w") as f:
        json.dump(cameras, f, indent=4)
    print(f"Camera poses saved to {output_path}")

# Step 7: Generate Combined Point Cloud
def generate_combined_point_cloud(point_clouds, output_path):
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in point_clouds:
        combined_pcd += pcd
    # Optional: Downsample for efficiency
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Combined point cloud saved to {output_path}")

# Step 8: Generate Masks Using SAM2
def generate_masks(image_paths, masks_path, checkpoint, model_cfg):
    os.makedirs(masks_path, exist_ok=True)
    for image_path in image_paths:
        mask_output_path = os.path.join(masks_path, os.path.splitext(os.path.basename(image_path))[0] + '.png')
        generate_mask(image_path, mask_output_path, checkpoint, model_cfg)

def generate_mask(image_path, mask_output_path, checkpoint, model_cfg):
    # Load the SAM2 model
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    center_point = [[(width // 2 - width // 10), height // 2]]
    point_labels = [1]
    point_coords = torch.tensor(center_point, dtype=torch.float32).unsqueeze(0).cuda()
    point_labels = torch.tensor(point_labels, dtype=torch.int64).unsqueeze(0).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(point_coords, point_labels=point_labels)
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)
    binary_mask = (combined_mask * 255).astype(np.uint8)
    # Save the binary mask (object in white, background in black)
    cv2.imwrite(mask_output_path, binary_mask)

def remove_background(image_path, output_path, mask_output_path, checkpoint, model_cfg, resize_factor=None):
    # Load the SAM 2 model
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image. Please check the image path.")
    
    # Resize the image if resize_factor is provided
    if resize_factor:
        height, width = image.shape[:2]
        new_dimensions = (int(width * resize_factor), int(height * resize_factor))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a point in the center of the image
    height, width, _ = image_rgb.shape
    center_point = [[(width // 2 - width // 10), height // 2]]  # Adjusted to center point directly
    point_labels = [1]  # label 1 indicates the foreground object

    # Convert the point and label to tensors
    point_coords = torch.tensor(center_point, dtype=torch.float32).unsqueeze(0).cuda()
    point_labels = torch.tensor(point_labels, dtype=torch.int64).unsqueeze(0).cuda()

    # Perform the prediction
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(point_coords, point_labels=point_labels)

    # Combine masks with optional morphological operations
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)
    
    # Optionally apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Ensure the combined mask is in uint8
    combined_mask = (combined_mask * 255).astype(np.uint8)

    # Save the mask image
    mask_filename = os.path.basename(image_path)
    mask_output_path = os.path.join(mask_output_path, mask_filename)
    cv2.imwrite(mask_output_path, combined_mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    # Create a white background
    background = np.ones_like(image, dtype=np.uint8) * 255
    mask_inv = cv2.bitwise_not(combined_mask)
    background = cv2.bitwise_or(background, background, mask=mask_inv)

    # Combine the isolated object with the white background
    final_image = cv2.bitwise_or(result, background)

    # Save the resulting image
    cv2.imwrite(output_path, final_image)


# Main Execution
if __name__ == "__main__":
    # Paths and configurations
    image_source_path = '/ec/pdx/disks/il_ailabs/jmrojasc/truck_web_cam'
    scene_path = 'data/truck_web_cam'
    images_path = os.path.join(scene_path, 'images')
    masks_path = os.path.join(scene_path, 'masks')
    os.makedirs(scene_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    matches_path = os.path.join(scene_path, 'matches')
    depth_maps_path = os.path.join(scene_path, 'depth_maps')
    os.makedirs(matches_path, exist_ok=True)
    os.makedirs(depth_maps_path, exist_ok=True)
    
    # Step 1: Copy images to scene path
    # image_files = sorted([x for x in os.listdir(image_source_path) if x.lower().endswith(('.jpg', '.jpeg', '.png'))])

    # Number of images you want to select
    num_desired_images = 4  # Adjust as needed

    # Step 1: Select images based on feature differences
    # pdb.set_trace()
    # selected_image_files, selected_indices = select_images_by_feature_difference(image_source_path, num_desired_images)
    selected_image_files = get_selected_images(scene_path, image_source_path, num_desired_images)
    # Copy selected images to scene path
    for img_file in selected_image_files:
        src_image_path = os.path.join(image_source_path, img_file)
        dst_image_path = os.path.join(images_path, img_file)
        image = cv2.imread(src_image_path)
        if image is None:
            print(f"Error: Unable to load image at {src_image_path}")
            continue
        cv2.imwrite(dst_image_path, image)
    
    # Step 2: Generate masks
    checkpoint = "/ec/pdx/disks/il_ailabs/ldgomezr/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    image_paths = [os.path.join(images_path, img_file) for img_file in selected_image_files]
    generate_masks(image_paths, masks_path, checkpoint, model_cfg)

    # remove background
    rem_bg = True
    if rem_bg:
        for image_path in image_paths:
            remove_background(image_path, image_path, image_path, checkpoint, model_cfg)
    
    # Step 3: Load images and masks
    images, masks, image_files = load_images_and_masks(images_path, masks_path)
    
    # Step 4: Initialize Depth Estimator
    depth_estimator, device = initialize_depth_estimator(encoder='vits', checkpoint_path='/ec/pdx/disks/il_ailabs/ldgomezr/Depth-Anything-V2/depth_anything_v2_vits.pth')
    
    # Step 5: Generate depth maps
    # pdb.set_trace()
    depth_maps = generate_depth_maps(images, masks, depth_estimator, device)
    
    # Step 6: Camera intrinsics (must be calibrated for your camera)
    fx = fy = 3196.91162109375  # Example focal length from your JSON
    cx = images[0].shape[1] / 2
    cy = images[0].shape[0] / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    
    # Step 7: Generate point clouds
    # pdb.set_trace()
    # Step 7: Estimate poses using feature matching and 3D-3D correspondences
    object_poses = estimate_poses(images, depth_maps, K, scene_path)

    
    # Step 9: Invert poses to get camera poses
    # pdb.set_trace()
    camera_poses = invert_poses(object_poses)
    
    # Step 10: Generate camera poses JSON
    # pdb.set_trace()
    json_output_path = os.path.join(scene_path, f'dust3r_{num_desired_images}.json')
    generate_camera_poses_json(camera_poses, image_files, images, K, json_output_path)

    # Step 10: Generate point clouds
    point_clouds = generate_point_clouds(images, masks, depth_maps, K)
    
    # Step 11: Generate combined point cloud
    # pdb.set_trace()
    point_cloud_output_path = os.path.join(scene_path, f'dust3r_{num_desired_images}.ply')
    generate_combined_point_cloud(point_clouds, point_cloud_output_path)
