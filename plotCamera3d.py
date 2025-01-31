import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from PIL import Image
import open3d as o3d

def load_camera_data(json_file):
    """
    Load camera data from a JSON file.

    Parameters:
    - json_file: Path to the JSON file containing camera data.

    Returns:
    - camera_data: List of dictionaries containing camera information.
    """
    with open(json_file, 'r') as f:
        camera_data = json.load(f)
    return camera_data

def plot_camera_poses_and_objects(camera_data, scene_path, output_file):
    """
    Plot camera poses and directions in a 3D space, overlay reference images, and add PLY objects to the plot.

    Parameters:
    - camera_data: List of dictionaries containing camera information.
    - scene_path: Path to the directory containing the reference images and PLY files.
    - output_file: Path to the output PNG file.
    """
    num_cameras = len(camera_data)
    fig = plt.figure(figsize=(500, 40))
    gs = GridSpec(6, num_cameras, height_ratios=[1, 2, 2, 2, 2, 2])

    # Create subplots for the reference images and their corresponding 3D views
    for i, camera in enumerate(camera_data):
        # Subplot for the reference image
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.axis('off')  # Hide the axes

        # Load and plot the reference image
        img_path = os.path.join(os.path.join(scene_path, 'images'), camera['img_name'])
        img = Image.open(img_path)
        img = np.array(img)

        ax_img.imshow(img, aspect='auto')
        ax_img.set_title(f"ID: {camera['id']} - {camera['img_name']}", fontsize=12)

        # Subplot for the individual 3D view
        ax_3d_individual = fig.add_subplot(gs[1, i], projection='3d')

        for other_camera in camera_data:
            position = np.array(other_camera['position'])
            rotation = np.array(other_camera['rotation'])

            # Plot camera position
            ax_3d_individual.scatter(position[0], position[1], position[2], c='r', marker='o')

            # Plot camera direction vector
            direction = rotation[:, 2]  # Assuming the third column is the forward direction
            ax_3d_individual.quiver(position[0], position[1], position[2], 
                                    direction[0], direction[1], direction[2], 
                                    length=0.5, color='b')

            # Annotate camera ID
            ax_3d_individual.text(position[0], position[1], position[2], f"ID: {other_camera['id']}", color='black')

        # Set the view to be behind the camera's perspective
        camera_position = np.array(camera['position'])
        camera_direction = -np.array(camera['rotation'])[:, 2]  # Invert the forward direction to get the backward direction
        elev = np.degrees(np.arctan2(camera_direction[2], np.sqrt(camera_direction[0]**2 + camera_direction[1]**2)))
        azim = np.degrees(np.arctan2(camera_direction[1], camera_direction[0]))
        ax_3d_individual.view_init(elev=elev, azim=azim)
        ax_3d_individual.set_title(f'Camera {camera["id"]} View')
        ax_3d_individual.set_xlabel('X')
        ax_3d_individual.set_ylabel('Y')
        ax_3d_individual.set_zlabel('Z')

    # Subplots for the dust3r_4.ply object (green)
    ax_3d_main_1 = fig.add_subplot(gs[2, :], projection='3d')
    ax_3d_main_2 = fig.add_subplot(gs[2, :num_cameras//2], projection='3d')
    ax_3d_main_3 = fig.add_subplot(gs[2, num_cameras//2:], projection='3d')

    ply_path = os.path.join(scene_path, 'dust3r_4.ply')
    if os.path.exists(ply_path):
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        ax_3d_main_1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='green', alpha=0.5, label='dust3r_4')
        ax_3d_main_2.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='green', alpha=0.5, label='dust3r_4')
        ax_3d_main_3.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='green', alpha=0.5, label='dust3r_4')

    ax_3d_main_1.set_title('dust3r_4 - View 1')
    ax_3d_main_1.set_xlabel('X')
    ax_3d_main_1.set_ylabel('Y')
    ax_3d_main_1.set_zlabel('Z')
    ax_3d_main_1.view_init(elev=30, azim=30)

    ax_3d_main_2.set_title('dust3r_4 - View 2')
    ax_3d_main_2.set_xlabel('X')
    ax_3d_main_2.set_ylabel('Y')
    ax_3d_main_2.set_zlabel('Z')
    ax_3d_main_2.view_init(elev=0, azim=90)  # Side view

    ax_3d_main_3.set_title('dust3r_4 - View 3')
    ax_3d_main_3.set_xlabel('X')
    ax_3d_main_3.set_ylabel('Y')
    ax_3d_main_3.set_zlabel('Z')
    ax_3d_main_3.view_init(elev=90, azim=0)  # Top-down view

    # Subplots for the point_cloud.ply object (blue)
    ax_3d_main_4 = fig.add_subplot(gs[3, :], projection='3d')
    ax_3d_main_5 = fig.add_subplot(gs[3, :num_cameras//2], projection='3d')
    ax_3d_main_6 = fig.add_subplot(gs[3, num_cameras//2:], projection='3d')

    point_cloud_path = os.path.join('output/gs_init/tt360/point_cloud/iteration_10000', 'point_cloud.ply')
    if os.path.exists(point_cloud_path):
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        points = np.asarray(point_cloud.points)
        ax_3d_main_4.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.5, label='Point Cloud')
        ax_3d_main_5.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.5, label='Point Cloud')
        ax_3d_main_6.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.5, label='Point Cloud')

    ax_3d_main_4.set_title('Point Cloud - View 1')
    ax_3d_main_4.set_xlabel('X')
    ax_3d_main_4.set_ylabel('Y')
    ax_3d_main_4.set_zlabel('Z')
    ax_3d_main_4.view_init(elev=30, azim=30)

    ax_3d_main_5.set_title('Point Cloud - View 2')
    ax_3d_main_5.set_xlabel('X')
    ax_3d_main_5.set_ylabel('Y')
    ax_3d_main_5.set_zlabel('Z')
    ax_3d_main_5.view_init(elev=0, azim=90)  # Side view

    ax_3d_main_6.set_title('Point Cloud - View 3')
    ax_3d_main_6.set_xlabel('X')
    ax_3d_main_6.set_ylabel('Y')
    ax_3d_main_6.set_zlabel('Z')
    ax_3d_main_6.view_init(elev=90, azim=0)  # Top-down view

    # Save the plot as a PNG file
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Camera poses plot saved to {output_file}")

if __name__ == "__main__":
    scene_path = 'data/tt360'
    json_file = os.path.join(scene_path, 'dust3r_50.json')  # Change this to the path of your .json file
    output_file = os.path.join(scene_path, 'camera_poses_50.png')  # Path to save the output PNG file

    camera_data = load_camera_data(json_file)
    plot_camera_poses_and_objects(camera_data, scene_path, output_file)
