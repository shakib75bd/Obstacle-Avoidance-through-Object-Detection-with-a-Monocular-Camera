# Road Navigation Evolution Tracking System

This enhanced road navigation system now includes comprehensive evolution matrices and performance tracking capabilities. The system automatically generates detailed reports with FPS metrics, angle accuracy analysis, and timeframe accuracy measurements.

## New Features

### 🚀 Evolution Tracking

- **Real-time Performance Monitoring**: Tracks FPS, processing times, and angle predictions
- **Accuracy Analysis**: Compares predictions against ground truth data
- **Evolution Matrices**: Multi-dimensional performance analysis
- **Comprehensive Reports**: Detailed text reports with statistics and recommendations

### 📊 Generated Reports Include:

- Average FPS and stability metrics
- Angle prediction accuracy percentages
- Processing time efficiency scores
- Obstacle detection statistics
- Performance evolution matrices
- Visual plots and charts
- System recommendations

## Files Structure

```
Depth-Codes/
├── roadmaxAngle.py          # Enhanced main navigation system
├── evolution_tracker.py     # Evolution tracking and analysis module
├── demo_evolution.py        # Demo script for testing
├── test_evolution.py        # Unit test with synthetic data
├── requirements.txt         # Python dependencies
├── evolutions/              # Generated reports directory
│   ├── ground_truth_sample.json  # Sample ground truth data
│   └── report-YYYYMMDD_HHMMSS.txt # Generated reports
└── README_Evolution.md      # This file
```

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have the following packages:
   - opencv-python
   - torch
   - torchvision
   - Pillow
   - numpy
   - matplotlib
   - seaborn (optional, for better plots)

## Usage

### Method 1: Use the Enhanced Main Script

```python
from roadmaxAngle import RoadNavigationSystem

# Initialize with evolution tracking enabled
navigation_system = RoadNavigationSystem(
    model_type="MiDaS_small",
    input_size=(128, 128),
    depth_threshold_factor=0.5,
    enable_evolution_tracking=True  # Enable tracking
)

# Run on video file or webcam
navigation_system.run("path1.mp4")  # or use 0 for webcam
```

### Method 2: Run the Demo Script

```bash
python demo_evolution.py
```

### Method 3: Test with Synthetic Data

```bash
python test_evolution.py
```

## Configuration Options

### RoadNavigationSystem Parameters:

- `model_type`: MiDaS model variant ("MiDaS_small", "DPT_Large", etc.)
- `input_size`: Input image size tuple (width, height)
- `depth_threshold_factor`: Sensitivity for obstacle detection
- `enable_evolution_tracking`: Enable/disable evolution tracking

### EvolutionTracker Parameters:

- `evolution_dir`: Directory for saving reports (default: "evolutions")

## Ground Truth Data

You can provide ground truth data for more accurate analysis:

1. Create a JSON file with expected angles and processing times:

```json
{
  "angles": {
    "1": 0.0,
    "2": -5.2,
    "3": 12.1
  },
  "timeframes": {
    "1": 0.033,
    "2": 0.033,
    "3": 0.033
  }
}
```

2. Use it in your code:

```python
tracker = EvolutionTracker()
tracker.set_ground_truth("path/to/ground_truth.json")
```

## Generated Report Contents

### Performance Metrics:

- **FPS Analysis**: Average, min, max, stability score
- **Angle Accuracy**: Mean error, standard deviation, accuracy percentage
- **Timeframe Accuracy**: Processing efficiency, real-time capability

### Evolution Matrices:

- **Performance Matrix**: FPS score, efficiency, angle accuracy, real-time capability
- **Accuracy Matrix**: Angle accuracy, time accuracy, detection rate
- **Efficiency Matrix**: Resource usage, speed, stability

### Visual Reports:

- FPS performance over time
- Turning angle distribution
- Processing time trends
- Performance evolution heatmaps
- Rolling accuracy trends
- System efficiency radar charts

## Key Metrics Explained

### FPS Metrics:

- **Average FPS**: Mean frames per second
- **FPS Stability**: Consistency of performance (0-100 score)
- **Real-time Capability**: Percentage of frames processed within real-time constraints

### Angle Accuracy:

- **Mean Error**: Average deviation from ground truth angles
- **Accuracy Percentage**: Frames within acceptable error range (±10°)

### Evolution Matrices:

- **Performance Matrix**: Normalized scores (0-100) for key performance indicators
- **Accuracy Matrix**: Precision metrics for predictions
- **Efficiency Matrix**: Resource utilization and system efficiency

## Report Location

Reports are automatically saved to the `evolutions/` directory with timestamps:

- Text report: `report-YYYYMMDD_HHMMSS.txt`
- Plot visualizations: `report-YYYYMMDD_HHMMSS_plots.png`

## Sample Output

```
========================================
ROAD NAVIGATION SYSTEM EVOLUTION REPORT
========================================
Generated: 2025-08-03 14:30:25
Session Duration: 45.67 seconds
Total Frames Processed: 150
Total Obstacles Detected: 23

PERFORMANCE METRICS:
- Average FPS: 28.45
- Angle Accuracy: 87.3%
- Real-time Capability: 94.2%

EVOLUTION MATRICES:
Performance Matrix (0-100 scale):
┌────────────────────────────────────────────────────────┐
│ FPS Score            │    85.20 │
│ Efficiency           │    92.15 │
│ Angle Accuracy       │    87.30 │
│ Real-time           │    94.20 │
└────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Common Issues:

1. **Import Error for seaborn**:

   - The system works without seaborn but with reduced plot quality
   - Install with: `pip install seaborn`

2. **Video file not found**:

   - Ensure `path1.mp4` exists or use webcam input (set video_path = 0)

3. **Low FPS performance**:

   - Try smaller input_size: `(64, 64)` or `(96, 96)`
   - Use lighter model: `"MiDaS_small"`

4. **Memory issues**:
   - Process fewer frames: increase `PROCESS_EVERY_N_FRAMES`
   - Reduce input resolution

## Advanced Usage

### Custom Ground Truth Generation:

```python
# Generate ground truth programmatically
ground_truth = {
    "angles": {str(i): expected_angle_for_frame_i for i in range(100)},
    "timeframes": {str(i): 0.033 for i in range(100)}  # 30 FPS ideal
}

import json
with open("custom_ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)
```

### Batch Processing:

```python
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in videos:
    system = RoadNavigationSystem(enable_evolution_tracking=True)
    system.run(video)
```

### Performance Optimization:

```python
# For maximum speed
system = RoadNavigationSystem(
    model_type="MiDaS_small",
    input_size=(64, 64),  # Smaller input
    enable_evolution_tracking=False  # Disable tracking for speed
)

# For maximum accuracy
system = RoadNavigationSystem(
    model_type="DPT_Large",
    input_size=(256, 256),  # Higher resolution
    enable_evolution_tracking=True
)
```

## Contributing

To extend the evolution tracking system:

1. Add new metrics to `EvolutionTracker.record_frame_metrics()`
2. Implement calculation methods following the pattern of existing metrics
3. Update the report generation to include new metrics
4. Add corresponding visualizations in `save_evolution_plots()`

The system is designed to be modular and extensible for additional performance metrics and analysis capabilities.
