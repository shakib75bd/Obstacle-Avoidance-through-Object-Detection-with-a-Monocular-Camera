import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import time

# Check if MPS, CUDA, or CPU is available
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MiDaS model
model_type = "MiDaS_small"  # Use smaller model for faster processing
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()

# Define transformations
transform = Compose([
    Resize((128, 128)),  # Further reduce resolution for faster processing
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_depth(input_image):
    """Get depth estimation for an input image."""
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = midas(input_tensor).squeeze().cpu().numpy()
    return depth

def determine_obstacles(depth_map, min_object_size=2000, depth_threshold_factor=0.5, confidence_threshold=0.1):
    """
    Determine which side (left, center, right) has the clearest path.
    Prioritizes the center direction if weights are too close.
    """
    height, width = depth_map.shape
    third_width = width // 3

    # Calculate dynamic depth threshold based on mean and standard deviation
    depth_mean = np.mean(depth_map)
    depth_std = np.std(depth_map)
    dynamic_threshold = depth_mean - depth_threshold_factor * depth_std

    # Identify significant obstacles based on dynamic threshold
    significant_obstacles = depth_map < dynamic_threshold
    significant_obstacles = significant_obstacles.astype(np.uint8) * 255

    # Filter out small objects using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    significant_obstacles = cv2.morphologyEx(significant_obstacles, cv2.MORPH_OPEN, kernel)

    # Split depth map into left, center, and right thirds
    left_region = significant_obstacles[:, :third_width]
    center_region = significant_obstacles[:, third_width:2*third_width]
    right_region = significant_obstacles[:, 2*third_width:]

    # Calculate weights for each region (lower weight = fewer obstacles)
    left_weight = np.sum(left_region) / left_region.size
    center_weight = np.sum(center_region) / center_region.size
    right_weight = np.sum(right_region) / right_region.size

    # Debugging: Print weights for each region
    print(f"Weights - Left: {left_weight}, Center: {center_weight}, Right: {right_weight}")

    # Find the minimum weight (clearest path)
    min_weight = min(left_weight, center_weight, right_weight)

    # Determine if the weights are too close
    if abs(min_weight - left_weight) < confidence_threshold and \
       abs(min_weight - center_weight) < confidence_threshold and \
       abs(min_weight - right_weight) < confidence_threshold:
        return "center"  # Default to center if weights are too close

    # Otherwise, choose the direction with the minimum weight
    if min_weight == center_weight:
        return "center"
    elif min_weight == left_weight:
        return "left"
    else:
        return "right"

# Load video file
video_path = "road_video.mp4"  # local video file or camcorder
cap = cv2.VideoCapture(video_path)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Convert frame to PIL image and get depth
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_map = get_depth(input_image)

        # Determine obstacle-rich direction
        direction = determine_obstacles(depth_map)

        # Draw direction on frame
        if direction == "left":
            cv2.arrowedLine(frame, (150, frame.shape[0]//2), (50, frame.shape[0]//2), (0, 0, 255), 5, tipLength=0.5)
        elif direction == "center":
            # Draw a straight-up arrow for "center"
            cv2.arrowedLine(frame,
                            (frame.shape[1]//2, frame.shape[0] - 50),  # Start point (bottom center)
                            (frame.shape[1]//2, frame.shape[0] - 150), # End point (top center)
                            (0, 0, 255), 5, tipLength=0.5)
        else:  # direction == "right"
            cv2.arrowedLine(frame, (frame.shape[1] - 150, frame.shape[0]//2),
                            (frame.shape[1] - 50, frame.shape[0]//2), (0, 0, 255), 5, tipLength=0.5)

        # Display frame
        cv2.imshow("Depth and Obstacle Detection", frame)

        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        print("FPS:", fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
