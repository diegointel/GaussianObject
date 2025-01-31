import os
import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Step 1: Extract camera poses from images in Image Source
def extract_camera_poses(image_source_path, num_images):
    image_files = sorted(os.listdir(image_source_path))
    images = [cv2.imread(os.path.join(image_source_path, img_file), cv2.IMREAD_GRAYSCALE) for img_file in image_files[:num_images]]
    
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptors_list = []
    
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
    
    poses = [np.eye(4)]
    
    for i in range(1, num_images):
        if descriptors_list[i-1] is None or descriptors_list[i] is None:
            print(f"Skipping image pair {i-1} and {i} due to missing descriptors.")
            continue
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_list[i-1], descriptors_list[i])
        matches = sorted(matches, key=lambda x: x.distance)
        
        src_pts = np.float32([keypoints_list[i-1][m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints_list[i][m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        
        poses.append(pose @ poses[-1])
    
    return poses, image_files

# Step 2: Select equidistant images
def select_equidistant_images(poses, sparse_num):
    step = len(poses) // sparse_num
    selected_indices = [i * step for i in range(sparse_num)]
    
    return selected_indices

# Step 3: Copy selected images from Image Source to Scene Path
def copy_selected_images(image_source_path, scene_path, image_files, selected_indices):
    images_path = os.path.join(scene_path, 'images')
    os.makedirs(images_path, exist_ok=True)
    
    copied_files = []
    for idx in selected_indices:
        image_file = image_files[idx]
        src_image_path = os.path.join(image_source_path, image_file)
        dst_image_path = os.path.join(images_path, image_file)
        
        image = cv2.imread(src_image_path)
        if image is None:
            print(f"Error: Unable to load image at {src_image_path}")
            continue
        cv2.imwrite(dst_image_path, image)
        copied_files.append(dst_image_path)  # Keep track of copied images for mask generation
    
    return copied_files

# Step 4: Generate masks for copied images (object in white, background in black)
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
    center_point = [[width // 2, height // 2]]
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

# Helper function to remove background and save result in the images folder
def remove_background(image_path, output_path, mask_output_path, checkpoint, model_cfg):
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    center_point = [[width // 2, height // 2]]
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
    
    # Apply the binary mask to create the image with the background removed
    result = cv2.bitwise_and(image, image, mask=binary_mask)
    
    # Save the resulting image with background removed
    cv2.imwrite(output_path, result)

# Main execution
if __name__ == "__main__":
    image_source_path = '/ec/pdx/disks/il_ailabs/jmrojasc/truck360b'
    sparse_num = 8
    num_images = len(os.listdir(image_source_path))
    scene_path = 'data/truck360b'
    images_path = f'{scene_path}/images'
    masks_path = f'{scene_path}/masks'
    os.makedirs(scene_path, exist_ok=True)
    
    # Step 1: Extract camera poses from Image Source
    poses, image_files = extract_camera_poses(image_source_path, num_images)
    
    # Step 2: Select equidistant images
    selected_indices = select_equidistant_images(poses, sparse_num)
    
    # Step 3: Copy selected images from Image Source to Scene Path
    copied_files = copy_selected_images(image_source_path, scene_path, image_files, selected_indices)
    
    # Step 4: Generate masks and optionally remove background
    remove_bg = True  # Set this dynamically as needed
    checkpoint = "/ec/pdx/disks/il_ailabs/ldgomezr/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    
    for image_path in copied_files:
        # Define paths for the mask and the output image with background removed
        mask_output_path = os.path.join(masks_path, os.path.basename(image_path))
        output_path = image_path  # Save background-removed image in the images folder

        # Generate the mask with object in white and background in black
        generate_mask(image_path, mask_output_path, checkpoint, model_cfg)

        # If remove_bg is enabled, create and save the image with background removed
        if remove_bg:
            remove_background(image_path, output_path, mask_output_path, checkpoint, model_cfg)
