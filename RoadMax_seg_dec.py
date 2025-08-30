
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from evolution_tracker import EvolutionTracker

# For YOLOv8, use Ultralytics' official package
def load_yolov8():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install the 'ultralytics' package: pip install ultralytics")
    model = YOLO('yolov8n.pt')  # Use nano model for speed; can change to yolov8s.pt, etc.
    return model


# Real-time optimized version with lightweight segmentation and angle output

class RoadMaxSegDecFast:
    def __init__(self, device=None, input_size=(320, 192), enable_evolution_tracking=True):
        self.device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = input_size
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        self.segmentation_model = deeplabv3_mobilenet_v3_large(pretrained=True).to(self.device).eval()
        self.segmentation_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.yolo_model = load_yolov8()
        self.enable_evolution_tracking = enable_evolution_tracking
        if self.enable_evolution_tracking:
            self.evolution_tracker = EvolutionTracker(evolution_dir="evolutions_seg_dec")

    def segment_road(self, frame):
        input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.segmentation_transform(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)['out'][0]
        seg_mask = output.argmax(0).cpu().numpy()
        drivable_mask = (seg_mask != 0).astype(np.uint8) * 255
        drivable_mask = cv2.resize(drivable_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return drivable_mask

    def detect_objects(self, frame):
        results = self.yolo_model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
        classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0].boxes, 'cls') else []
        return detections, classes

    def calculate_angle(self, drivable_mask):
        h, w = drivable_mask.shape
        bottom = drivable_mask[int(h*0.7):, :]
        M = cv2.moments(bottom)
        if M["m00"] == 0:
            return 0  # No drivable area detected, go straight
        cx = int(M["m10"] / M["m00"])
        frame_center = w // 2
        offset = cx - frame_center
        max_offset = w // 2
        angle = (offset / max_offset) * 45  # Scale to +/- 45 degrees
        return angle

    def process_frame(self, frame, frame_number):
        start_time = time.time()
        small_frame = cv2.resize(frame, self.input_size)
        drivable_mask = self.segment_road(small_frame)
        detections, classes = self.detect_objects(small_frame)
        angle = self.calculate_angle(drivable_mask)
        # Visualization (optional, can be commented out for speed)
        overlay = small_frame.copy()
        overlay[drivable_mask == 0] = (0, 0, 100)
        alpha = 0.4
        vis_frame = cv2.addWeighted(overlay, alpha, small_frame, 1 - alpha, 0)
        for box, cls in zip(detections, classes):
            x1, y1, x2, y2 = map(int, box)
            obj_mask = drivable_mask[y1:y2, x1:x2]
            color = (0, 255, 0) if np.any(obj_mask) else (0, 0, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, f"Obj {int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw angle arrow
        h, w = vis_frame.shape[:2]
        center = (w // 2, h - 30)
        length = 60
        angle_rad = np.radians(angle)
        end_x = int(center[0] + length * np.sin(angle_rad))
        end_y = int(center[1] - length * np.cos(angle_rad))
        cv2.arrowedLine(vis_frame, center, (end_x, end_y), (255, 255, 0), 4, tipLength=0.4)
        cv2.putText(vis_frame, f"Angle: {angle:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        obstacles_detected = len(detections) > 0
        if self.enable_evolution_tracking:
            self.evolution_tracker.record_frame_metrics(
                frame_number, fps, angle, processing_time, obstacles_detected
            )
        return vis_frame, angle

    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)
        frame_count = 0
        processed_frame_count = 0
        try:
            print("Starting RoadMaxSegDecFast with evaluation...")
            print(f"Evolution tracking: {'Enabled' if self.enable_evolution_tracking else 'Disabled'}")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                processed_frame_count += 1
                vis_frame, angle = self.process_frame(frame, processed_frame_count)
                print(f"Frame {processed_frame_count}: Angle: {angle:.2f} degrees")
                cv2.imshow("RoadMax Fast: Segmentation & Detection", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.enable_evolution_tracking and processed_frame_count > 0:
                print("\nGenerating evolution report...")
                report_path = self.evolution_tracker.generate_report()
                print(f"Evolution report saved to: {report_path}")
                fps_metrics = self.evolution_tracker.calculate_fps_metrics()
                angle_metrics = self.evolution_tracker.calculate_angle_accuracy()
                print(f"\nSession Summary:")
                print(f"- Processed {processed_frame_count} frames")
                print(f"- Average FPS: {fps_metrics['average_fps']:.2f}")
                print(f"- Angle Accuracy: {angle_metrics['accuracy_percentage']:.2f}%")
                print(f"- Obstacles Detected: {sum(self.evolution_tracker.obstacle_detection_data)}")
            else:
                print("No frames processed or evolution tracking disabled.")

if __name__ == "__main__":
    video_path = "path1.mp4"  # Change as needed
    system = RoadMaxSegDecFast()
    system.run(video_path)
