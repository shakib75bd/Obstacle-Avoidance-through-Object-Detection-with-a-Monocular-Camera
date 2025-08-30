import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

def load_yolov8():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install the 'ultralytics' package: pip install ultralytics")
    model = YOLO('yolov8n.pt')
    return model

def load_midas(device):
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device).eval()
    # Use smaller input size for faster inference
    transform = Compose([
        Resize((128, 128)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return midas, transform

class RoadMaxYoloDepth:
    def __init__(self, grid_cols=6, grid_rows=3, show_visualization=True):
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = load_yolov8()
        self.midas, self.midas_transform = load_midas(self.device)
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.show_visualization = show_visualization

    def get_depth(self, frame):
        # Downscale frame for depth estimation
        small = cv2.resize(frame, (128, 128))
        input_image = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        input_tensor = self.midas_transform(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth = self.midas(input_tensor).squeeze().cpu().numpy()
        # Upscale back to frame size
        depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        return depth_resized

    def detect_objects(self, frame):
        results = self.yolo(frame)
        detections = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
        return detections

    def get_grid_boxes(self, frame):
        h, w = frame.shape[:2]
        # Focus on bottom 2/3 of the frame
        y_start = h // 3
        y_end = h
        cell_w = w // self.grid_cols
        cell_h = (y_end - y_start) // self.grid_rows
        boxes = []
        for i in range(self.grid_cols):
            for j in range(self.grid_rows):
                x1 = i * cell_w
                y1 = y_start + j * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                boxes.append((x1, y1, x2, y2))
        return boxes

    def get_box_density_and_depth(self, boxes, detections, depth_map):
        densities = []
        depths = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Density: calculate both count and area coverage of detections
            count = 0
            area_coverage = 0
            cell_area = (x2 - x1) * (y2 - y1)
            for det in detections:
                ix1 = max(x1, det[0])
                iy1 = max(y1, det[1])
                ix2 = min(x2, det[2])
                iy2 = min(y2, det[3])
                if ix1 < ix2 and iy1 < iy2:
                    count += 1
                    overlap_area = (ix2 - ix1) * (iy2 - iy1)
                    area_coverage += overlap_area / cell_area
            # Combine count and area coverage for density score
            density_score = count + 2 * area_coverage  # Weight area coverage more
            densities.append(density_score)

            # Depth: use both median and minimum depth for better obstacle avoidance
            cell_depths = depth_map[y1:y2, x1:x2]
            median_depth = np.median(cell_depths)
            min_depth = np.percentile(cell_depths, 10)  # 10th percentile for robustness
            # Combine depths with weight towards minimum (closer obstacles)
            combined_depth = 0.3 * median_depth + 0.7 * min_depth
            depths.append(combined_depth)
        return densities, depths

    def select_optimal_box(self, boxes, densities, depths, frame):
        # Try 3x3 region first, then fallback to 2x2; allow regions with lowest total density
        h, w = frame.shape[:2]
        cols, rows = self.grid_cols, self.grid_rows
        density_grid = np.array(densities).reshape((cols, rows)).T  # shape: (rows, cols)
        depth_grid = np.array(depths).reshape((cols, rows)).T
        best_score = float('-inf')
        best_center = None
        best_region = None
        region_size = None

        # Helper function to evaluate region safety
        def evaluate_region(region_density, region_depth):
            # Check if any cell has very high density (obstacle)
            if np.max(region_density) > 2.0:  # Threshold for clear obstacle
                return float('-inf')
            # Check if depth variation is too high (uneven terrain)
            if np.std(region_depth) > np.mean(region_depth) * 0.5:
                return float('-inf')
            return -np.sum(region_density) + 0.7 * np.mean(region_depth)

        # Try 3x3 regions
        for i in range(rows-2):
            for j in range(cols-2):
                region_density = density_grid[i:i+3, j:j+3]
                region_depth = depth_grid[i:i+3, j:j+3]
                center_idx = (i+1)*cols + (j+1)
                box = boxes[center_idx]
                cell_cx = (box[0] + box[2]) // 2
                cell_cy = (box[1] + box[3]) // 2

                # Distance from center-bottom of frame
                dist_from_center = abs(cell_cx - w//2)
                dist_from_bottom = h - cell_cy
                position_score = -(0.01 * dist_from_center + 0.005 * dist_from_bottom)

                # Evaluate region safety and traversability
                region_score = evaluate_region(region_density, region_depth)
                if region_score == float('-inf'):
                    continue

                # Combined score
                score = region_score + position_score

                if score > best_score:
                    best_score = score
                    best_center = (cell_cx, cell_cy, box)
                    best_region = (i, j, 3)
                    region_size = 3
        # If no 3x3 found, try 2x2
        for i in range(rows-1):
            for j in range(cols-1):
                region = density_grid[i:i+2, j:j+2]
                region_depth = depth_grid[i:i+2, j:j+2]
                total_density = np.sum(region)
                center_idx = (i+1)*cols + (j+1)
                box = boxes[center_idx]
                cell_cx = (box[0] + box[2]) // 2
                cell_cy = (box[1] + box[3]) // 2
                dist = np.hypot(cell_cx - w//2, h - cell_cy)
                center_depth = depth_grid[i+1, j+1]
                score = -total_density + 0.5 * center_depth - 0.01 * dist
                if score > best_score:
                    best_score = score
                    best_center = (cell_cx, cell_cy, box)
                    best_region = (i, j, 2)
                    region_size = 2
        if best_center is not None:
            return best_center, best_region, region_size
        return None, None, None

    def calculate_angle(self, frame, cell_cx):
        w = frame.shape[1]
        offset = cell_cx - (w // 2)
        max_offset = w // 2
        angle = (offset / max_offset) * 45  # Scale to +/- 45 degrees
        return angle

    def process_frame(self, frame):
        depth_map = self.get_depth(frame)
        detections = self.detect_objects(frame)
        boxes = self.get_grid_boxes(frame)
        densities, depths = self.get_box_density_and_depth(boxes, detections, depth_map)
        best, best_region, region_size = self.select_optimal_box(boxes, densities, depths, frame) if boxes else (None, None, None)
        if best:
            cell_cx, cell_cy, best_box = best
            angle = self.calculate_angle(frame, cell_cx)
        else:
            angle = 0
        if self.show_visualization:
            vis = frame.copy()
            # Draw grid
            for box, density, depth in zip(boxes, densities, depths):
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 2)
                cv2.putText(vis, f"D:{density}", (x1+2, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                cv2.putText(vis, f"Z:{depth:.2f}", (x1+2, y1+36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,0,255), 2)
            # Highlight the selected region (3x3 or 2x2) as a solid block (all cells together)
            if best_region is not None:
                i, j, size = best_region
                cols, rows = self.grid_cols, self.grid_rows
                # Create a mask for the selected region
                region_mask = np.zeros(vis.shape[:2], dtype=np.uint8)
                for di in range(size):
                    for dj in range(size):
                        idx = (i+di)*cols + (j+dj)
                        x1, y1, x2, y2 = boxes[idx]
                        region_mask[y1:y2, x1:x2] = 1
                # Overlay the region with yellow
                overlay = vis.copy()
                overlay[region_mask == 1] = (0,255,255)
                alpha = 0.3
                vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
                # Draw a thick border around the entire region
                # Find the bounding box of the region
                region_coords = [boxes[(i+di)*cols + (j+dj)] for di in range(size) for dj in range(size)]
                x1s, y1s, x2s, y2s = zip(*region_coords)
                rx1, ry1, rx2, ry2 = min(x1s), min(y1s), max(x2s), max(y2s)
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0,255,255), 4)
            if best:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 3)
            cv2.putText(vis, f"Angle: {angle:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return vis, angle
        else:
            return None, angle

    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            vis, angle = self.process_frame(frame)
            print(f"Steering Angle: {angle:.2f} degrees")
            if self.show_visualization and vis is not None:
                cv2.imshow("RoadMax YOLO+Depth", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        if self.show_visualization:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "path1.mp4"  # Change as needed
    # Divide bottom 2/3 into 20 sub-boxes (5 cols x 4 rows) for finer granularity
    system = RoadMaxYoloDepth(grid_cols=5, grid_rows=4, show_visualization=True)
    system.run(video_path)
