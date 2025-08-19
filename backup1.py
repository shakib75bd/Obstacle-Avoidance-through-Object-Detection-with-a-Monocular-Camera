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
model_type = "DPT_Hybrid"  # Use "MiDaS_small" for faster processing
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()

# Define transformations
transform = Compose([
    Resize((256, 256)),  # Resize for faster processing
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_depth(input_image):
    """Get depth estimation for an input image."""
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = midas(input_tensor).squeeze().cpu().numpy()
    return depth

def determine_obstacles(depth_map, depth_threshold_factor=0.5):
    """
    Calculate the turning angle based on the distribution of obstacles in the depth map.
    Returns the turning angle in degrees (negative for left, positive for right, 0 for straight).
    """
    height, width = depth_map.shape

    # Normalize the depth map to ensure consistent scaling
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Calculate dynamic depth threshold based on mean and standard deviation
    depth_mean = np.mean(depth_map_normalized)
    depth_std = np.std(depth_map_normalized)
    dynamic_threshold = depth_mean - depth_threshold_factor * depth_std

    # Identify significant obstacles based on dynamic threshold
    significant_obstacles = depth_map_normalized < dynamic_threshold
    significant_obstacles = significant_obstacles.astype(np.uint8) * 255

    # Debugging: Display the obstacle map
    cv2.imshow("Obstacle Map", significant_obstacles)

    # Calculate the center of mass of obstacles
    moments = cv2.moments(significant_obstacles)
    if moments["m00"] == 0:
        print("No obstacles detected. Going straight.")
        return 0  # No obstacles, go straight

    # Calculate the horizontal center of mass
    center_of_mass_x = int(moments["m10"] / moments["m00"])
    frame_center_x = width // 2
    offset_x = center_of_mass_x - frame_center_x

    # Debugging: Print center of mass and offset
    print(f"Center of Mass X: {center_of_mass_x}, Frame Center X: {frame_center_x}, Offset X: {offset_x}")

    # Calculate the turning angle based on the offset
    max_offset = width // 2  # Maximum possible offset
    turning_angle = (offset_x / max_offset) * 45  # Scale to +/- 45 degrees

    # Debugging: Print the turning angle
    print(f"Turning Angle: {turning_angle:.2f} degrees")

    return turning_angle

def draw_direction(frame, turning_angle):
    """
    Draw an arrow on the frame based on the turning angle.
    """
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Calculate the end point of the arrow based on the turning angle
    if turning_angle == 0:
        # Draw a straight-up arrow for "center"
        cv2.arrowedLine(frame,
                        (center_x, height - 50),  # Start point (bottom center)
                        (center_x, height - 150), # End point (top center)
                        (0, 0, 255), 5, tipLength=0.5)
    else:
        # Calculate the end point for left or right turn
        angle_rad = np.radians(turning_angle)
        arrow_length = 100
        end_x = int(center_x + arrow_length * np.sin(angle_rad))
        end_y = int(center_y - arrow_length * np.cos(angle_rad))
        cv2.arrowedLine(frame,
                        (center_x, center_y),  # Start point (center)
                        (end_x, end_y),       # End point (based on angle)
                        (0, 0, 255), 5, tipLength=0.5)

    # Display the turning angle on the frame
    cv2.putText(frame, f"Angle: {turning_angle:.2f} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Load webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Convert frame to PIL image and get depth
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_map = get_depth(input_image)

        # Resize depth map back to frame size for display
        depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        depth_grayscale = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Determine turning angle
        turning_angle = determine_obstacles(depth_resized)

        # Draw direction on frame
        draw_direction(frame, turning_angle)

        # Display depth map and frame
        cv2.imshow("Depth Map", depth_grayscale)
        cv2.imshow("Direction", frame)

        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        print("FPS:", fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
