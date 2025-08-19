import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import time
from evolution_tracker import EvolutionTracker

class RoadNavigationSystem:
    def __init__(self, model_type="MiDaS_small", input_size=(128, 128),
                 depth_threshold_factor=0.5, enable_evolution_tracking=True):
        # Initialize model and parameters
        self.model_type = model_type
        self.input_size = input_size
        self.depth_threshold_factor = depth_threshold_factor

        # Initialize evolution tracking
        self.enable_evolution_tracking = enable_evolution_tracking
        if self.enable_evolution_tracking:
            self.evolution_tracker = EvolutionTracker()

        self.setup_model()

    def setup_model(self):
        # Setup the model and device
        device = "mps" if torch.backends.mps.is_available() else \
                ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device).eval()

        # Define transformations
        self.transform = Compose([
            Resize(self.input_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_depth(self, input_image):
        """Get depth estimation for an input image."""
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.midas(input_tensor).squeeze().cpu().numpy()
        return depth

    def determine_obstacles(self, depth_map):
        # Add multiple regions of interest (ROI)
        height, width = depth_map.shape

        # Create a road ROI (focus on bottom half of image)
        roi_mask = np.zeros_like(depth_map)
        roi_mask[height//2:, :] = 1

        # Apply ROI to depth map
        depth_roi = depth_map * roi_mask

        # Use distance-weighted obstacle detection
        # Objects closer to the bottom of the frame have higher importance
        weight_map = np.zeros_like(depth_map)
        for y in range(height):
            # Weight increases as we move down the image
            weight = y / height
            weight_map[y, :] = weight

        # Apply weighting to depth map
        weighted_depth = depth_map * weight_map

        # Rest of your obstacle detection code...
        height, width = depth_map.shape

        # Normalize the depth map to ensure consistent scaling
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate dynamic depth threshold based on mean and standard deviation
        depth_mean = np.mean(depth_map_normalized)
        depth_std = np.std(depth_map_normalized)
        dynamic_threshold = depth_mean - self.depth_threshold_factor * depth_std

        # Identify significant obstacles based on dynamic threshold
        significant_obstacles = depth_map_normalized < dynamic_threshold
        significant_obstacles = significant_obstacles.astype(np.uint8) * 255

        # Debugging: Display the obstacle map
        cv2.imshow("Obstacle Map", significant_obstacles)

        # Calculate the center of mass of obstacles
        moments = cv2.moments(significant_obstacles)
        obstacles_detected = moments["m00"] > 0

        if moments["m00"] == 0:
            print("No obstacles detected. Going straight.")
            return 0, obstacles_detected  # No obstacles, go straight

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

        return turning_angle, obstacles_detected

    def process_frame(self, frame, frame_number):
        # Process a single frame
        frame_start_time = time.time()

        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_map = self.get_depth(input_image)

        # Resize depth map back to frame size for display
        depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        depth_grayscale = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Determine turning angle and obstacle detection
        turning_angle, obstacles_detected = self.determine_obstacles(depth_resized)

        # Calculate processing time and FPS
        processing_time = time.time() - frame_start_time
        fps = 1 / processing_time if processing_time > 0 else 0

        # Record metrics for evolution tracking
        if self.enable_evolution_tracking:
            self.evolution_tracker.record_frame_metrics(
                frame_number, fps, turning_angle, processing_time, obstacles_detected
            )

        # Draw direction on frame
        self.draw_direction(frame, turning_angle)

        # Display depth map and frame
        cv2.imshow("Depth Map", depth_grayscale)
        cv2.imshow("Direction", frame)

        return fps, turning_angle, processing_time

    def draw_direction(self, frame, turning_angle):
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

    def run(self, video_source):
        # Main processing loop
        cap = cv2.VideoCapture(video_source)
        frame_count = 0
        processed_frame_count = 0
        PROCESS_EVERY_N_FRAMES = 2

        try:
            print("Starting road navigation system...")
            print(f"Evolution tracking: {'Enabled' if self.enable_evolution_tracking else 'Disabled'}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue

                processed_frame_count += 1

                # Process frame and get metrics
                fps, turning_angle, processing_time = self.process_frame(frame, processed_frame_count)

                # Print performance metrics
                print(f"Frame {processed_frame_count}: FPS: {fps:.2f}, Angle: {turning_angle:.2f}°, Time: {processing_time:.4f}s")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Generate evolution report
            if self.enable_evolution_tracking and processed_frame_count > 0:
                print("\nGenerating evolution report...")
                report_path = self.evolution_tracker.generate_report()
                print(f"Evolution report saved to: {report_path}")

                # Print summary statistics
                fps_metrics = self.evolution_tracker.calculate_fps_metrics()
                angle_metrics = self.evolution_tracker.calculate_angle_accuracy()
                print(f"\nSession Summary:")
                print(f"- Processed {processed_frame_count} frames")
                print(f"- Average FPS: {fps_metrics['average_fps']:.2f}")
                print(f"- Angle Accuracy: {angle_metrics['accuracy_percentage']:.2f}%")
                print(f"- Obstacles Detected: {sum(self.evolution_tracker.obstacle_detection_data)}")
            else:
                print("No frames processed or evolution tracking disabled.")

# Load video file
video_path = "path1.mp4"  # Replace with your video file or camera input (0 for webcam)
navigation_system = RoadNavigationSystem()
navigation_system.run(video_path)
