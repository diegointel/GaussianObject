import os
import numpy as np
import cv2
import subprocess
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import math

# Step 1: Extract camera poses from images using feature matching and essential matrix estimation
def extract_camera_poses(scene_path, num_images):
    images_path = os.path.join(scene_path, 'images')
    image_files = sorted(os.listdir(images_path))
    
    # Load the images
    images = [cv2.imread(os.path.join(images_path, img_file), cv2.IMREAD_GRAYSCALE) for img_file in image_files[:num_images]]
    
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
    for i in range(1, num_images):
        # Match features between image i-1 and image i
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_list[i-1], descriptors_list[i])
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract points from the matches
        src_pts = np.float32([keypoints_list[i-1][m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_list[i][m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Recover relative camera pose from the essential matrix
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)
        
        # Compose the transformation matrix (4x4)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        
        # Append the new pose
        poses.append(pose @ poses[-1])  # Chain the poses
    
    return poses

# Step 2: Select equidistant images based on camera poses and ensure different selections each time
def select_equidistant_images(poses, sparse_num):
    # Calculate step size based on remaining poses
    step = len(poses) // sparse_num
    
    # Select images spaced by the step size
    selected_indices = [i * step for i in range(sparse_num)]
    
    # Remove selected poses from the poses list for next iteration
    remaining_poses = [pose for i, pose in enumerate(poses) if i not in selected_indices]
    
    return selected_indices, remaining_poses

# Step 3: Create sparse file for selected images (fixed as sparse_num.txt)
def modify_sparse_file(scene_path, selected_indices, sparse_num):
    sparse_file_path = os.path.join(scene_path, f'sparse_{sparse_num}.txt')  # Name depends on sparse_num
    with open(sparse_file_path, 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")
    return selected_indices

# Step 4: Execute pred_poses.py (always using sparse_num)
def execute_pred_poses(scene_path, sparse_num):
    command = f"python pred_poses.py -s {scene_path} --sparse_num {sparse_num}"  # Fixed to sparse_num
    subprocess.run(command, shell=True)

# Step 5: Create composite image based on sparse_num
def create_composite_image(scene_path, selected_indices, output_file, sparse_num):
    images_path = os.path.join(scene_path, 'images')
    selected_images = [Image.open(os.path.join(images_path, sorted(os.listdir(images_path))[idx])) for idx in selected_indices]
    
    # Determine grid size based on sparse_num
    cols = sparse_num
    rows = 1 if sparse_num <= 4 else 2
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    axs = axs.flatten() if sparse_num > 1 else [axs]
    
    for ax, img in zip(axs, selected_images):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Step 6: Capture 8 views from different angles using matplotlib (always 8 views)
def capture_view(mesh, elev, azim, output_file):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    vertices = np.asarray(mesh.vertices)
    
    if mesh.has_triangles():
        triangles = np.asarray(mesh.triangles)
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, alpha=0.6)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)
    
    ax.view_init(elev=elev, azim=azim)
    
    plt.savefig(output_file)
    plt.close()

# Step 7: Visualize 3D object and always save 8 views (fixed to 8 views)
def visualize_3d_object(scene_path, sparse_num, output_file, selected_indices):
    ply_file = os.path.join(scene_path, f'dust3r_{sparse_num}.ply')
    
    mesh = o3d.io.read_triangle_mesh(ply_file)
    
    if mesh.is_empty():
        print("[Open3D WARNING] The mesh does not contain any data.")
        return
    
    if not mesh.has_triangles():
        print("[Open3D WARNING] The file contains vertices but no triangles. Treating it as a point cloud.")
    
    # Always capture 8 views
    rotations = [(i * 360 // 8, i * 45) for i in range(8)]
    
    output_images = []
    
    for i, (elev, azim) in enumerate(rotations):
        view_file = os.path.join(scene_path, f'view_{i}.png')
        capture_view(mesh, elev, azim, view_file)
        output_images.append(view_file)
    
    images_path = os.path.join(scene_path, 'images')
    selected_images = [Image.open(os.path.join(images_path, sorted(os.listdir(images_path))[idx])) for idx in selected_indices]
    captured_images = [Image.open(img_file) for img_file in output_images]
    
    # Dynamically adjust the grid layout based on sparse_num (fixed 8 views for the mesh)
    total_images = sparse_num + 8  # sparse_num original images + 8 mesh views
    cols = min(4, total_images)  # Max 4 columns
    rows = math.ceil(total_images / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    axs = axs.flatten() if total_images > 1 else [axs]
    
    # Display original images
    for ax, img in zip(axs[:sparse_num], selected_images):
        ax.imshow(img)
        ax.axis('off')
    
    # Display captured views
    for ax, img in zip(axs[sparse_num:], captured_images):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
     # Rename the .ply file to include selected image indices
    indices_str = '_'.join(map(str, selected_indices))
    new_ply_file = os.path.join(scene_path, f'dust3r_{sparse_num}_indx_[{indices_str}].ply')
    
    # Rename the original ply file
    os.rename(ply_file, new_ply_file)
    print(f"Mesh file renamed to: {new_ply_file}")

# Main execution logic
if __name__ == "__main__":
    scene_path = 'data/lamp'
    sparse_num = 8  # Set this dynamically as needed (e.g., 4, 8, 2)
    num_images = len(os.listdir(os.path.join(scene_path, 'images')))
    
    # Step 1: Extract camera poses (for the entire set of images)
    poses = extract_camera_poses(scene_path, num_images)
    
    # Process 4 different sets of images
    for set_num in range(1, 2):
        print(f"Processing set {set_num}...")
        
        # Step 2: Select equidistant images and ensure they are different for each set
        selected_indices, poses = select_equidistant_images(poses, sparse_num)
        
        # Step 3: Modify sparse file for the selected indices
        modify_sparse_file(scene_path, selected_indices, sparse_num)
        
        # Step 4: Execute pred_poses.py using sparse_num
        execute_pred_poses(scene_path, sparse_num)
        
        # Step 5: Create composite image based on sparse_num
        composite_image_file = os.path.join(scene_path, f'composite_image_set_{set_num}.png')
        create_composite_image(scene_path, selected_indices, composite_image_file, sparse_num)
        
        # Step 6: Visualize 3D object and save views (always 8 views)
        output_3d_file = os.path.join(scene_path, f'3d_views_combined_set_{set_num}.png')
        visualize_3d_object(scene_path, sparse_num, output_3d_file, selected_indices)
        
        print(f"Set {set_num} processed: Composite image saved to {composite_image_file}, 3D views saved to {output_3d_file}")