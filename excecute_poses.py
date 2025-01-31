import os
import random
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

# Step 1: Modify sparse file
def modify_sparse_file(scene_path, num_images=4):
    images_path = os.path.join(scene_path, 'images')
    image_files = sorted(os.listdir(images_path))
    selected_indices = random.sample(range(len(image_files)), num_images)
    
    with open(os.path.join(scene_path, f'sparse_{num_images}.txt'), 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")
    
    return selected_indices

# Step 2: Read sparse file
def read_sparse_file(scene_path, num_images=4):
    sparse_file_path = os.path.join(scene_path, f'sparse_{num_images}.txt')
    
    if not os.path.exists(sparse_file_path):
        raise FileNotFoundError(f"The file {sparse_file_path} does not exist.")
    
    selected_indices = []
    with open(sparse_file_path, 'r') as f:
        for line in f:
            selected_indices.append(int(line.strip()))
    
    return selected_indices

# Step 3: Execute pred_poses.py (optional, depending on your pipeline)
def execute_pred_poses(scene_path, sparse_num):
    command = f"python pred_poses.py -s {scene_path} --sparse_num {sparse_num}"
    subprocess.run(command, shell=True)

# Step 4: Create composite image
def create_composite_image(scene_path, selected_indices, output_file):
    images_path = os.path.join(scene_path, 'images')
    selected_images = [Image.open(os.path.join(images_path, sorted(os.listdir(images_path))[idx])) for idx in selected_indices]
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for ax, img in zip(axs.flatten(), selected_images):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Step 5: Capture view from different angles using matplotlib
def capture_view(mesh, elev, azim, output_file):
    # Create an image with a matplotlib 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract vertices (and triangles if available)
    vertices = np.asarray(mesh.vertices)
    
    # If the mesh has no triangles, we'll plot the point cloud instead
    if mesh.has_triangles():
        triangles = np.asarray(mesh.triangles)
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, alpha=0.6)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1)
    
    # Rotate view: elev (rotation along x-axis), azim (rotation along z-axis)
    ax.view_init(elev=elev, azim=azim)
    
    # Save the view to a file
    plt.savefig(output_file)
    plt.close()

# Step 6: Visualize 3D object, save views, and combine with original images
def visualize_3d_object(scene_path, sparse_num, output_file, selected_indices):
    ply_file = os.path.join(scene_path, f'dust3r_{sparse_num}.ply')
    
    # Read the mesh or point cloud
    mesh = o3d.io.read_triangle_mesh(ply_file)
    
    # If it's empty, print a warning and return
    if mesh.is_empty():
        print("[Open3D WARNING] The mesh does not contain any data.")
        return
    
    # If it contains only vertices, treat it as a point cloud
    if not mesh.has_triangles():
        print("[Open3D WARNING] The file contains vertices but no triangles. Treating it as a point cloud.")
    
    # Capture views from different angles (azimuth and elevation)
    # Example: 8 different combinations of rotations around z-axis (azim) and x-axis (elev)
    rotations = [
        (0, 0), (30, 45), (30, 90), (30, 135), 
        (60, 180), (60, 225), (60, 270), (90, 315)
    ]  # (elev, azim)
    
    output_images = []
    
    for i, (elev, azim) in enumerate(rotations):
        view_file = os.path.join(scene_path, f'view_{i}.png')
        capture_view(mesh, elev, azim, view_file)
        output_images.append(view_file)
    
    # Read the original and captured 3D images
    images_path = os.path.join(scene_path, 'images')
    selected_images = [Image.open(os.path.join(images_path, sorted(os.listdir(images_path))[idx])) for idx in selected_indices]
    captured_images = [Image.open(img_file) for img_file in output_images]
    
    # Combine into a 3x4 grid (3 rows, 4 columns)
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))  # Changed to 3x4 grid
    
    # First row: Original images
    for ax, img in zip(axs[0], selected_images):
        ax.imshow(img)
        ax.axis('off')
    
    # Second row: First 4 captured images
    for ax, img in zip(axs[1], captured_images[:4]):
        ax.imshow(img)
        ax.axis('off')
    
    # Third row: Last 4 captured images
    for ax, img in zip(axs[2], captured_images[4:]):
        ax.imshow(img)
        ax.axis('off')
    
    # Save the combined image
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Main execution logic
if __name__ == "__main__":
    scene_path = 'data/truck_toy'
    sparse_num = 4
    
    # Step 1: Modify sparse file (optional, if you want to randomly select images)
    selected_indices = modify_sparse_file(scene_path, sparse_num)
    
    # Step 2: Read selected indices from the sparse file
    selected_indices = read_sparse_file(scene_path, sparse_num)
    
    # Step 3: Execute pred_poses.py (optional, if needed for your specific pipeline)
    execute_pred_poses(scene_path, sparse_num)
    
    # Step 4: Create composite image from selected images
    composite_image_file = os.path.join(scene_path, 'composite_image.png')
    create_composite_image(scene_path, selected_indices, composite_image_file)
    
    # Step 5: Visualize 3D object, save views, and combine with original images
    output_3d_file = os.path.join(scene_path, '3d_views_combined.png')
    visualize_3d_object(scene_path, sparse_num, output_3d_file, selected_indices)
    
    print(f"Composite image saved to {composite_image_file}")
    print(f"3D views saved to {output_3d_file}")
