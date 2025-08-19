import os
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Try to import seaborn, use matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not available, using matplotlib only for plots")

class EvolutionTracker:
    """
    Tracks and analyzes evolution metrics for road navigation system performance.
    Generates comprehensive reports with FPS, angle accuracy, and timeframe accuracy.
    """

    def __init__(self, evolution_dir: str = "evolutions"):
        self.evolution_dir = evolution_dir
        self.ensure_evolution_dir()

        # Performance metrics storage
        self.fps_data = []
        self.angle_data = []
        self.processing_times = []
        self.frame_timestamps = []
        self.obstacle_detection_data = []

        # Ground truth data (can be manually set or loaded from file)
        self.ground_truth_angles = {}  # frame_number: expected_angle
        self.ground_truth_timeframes = {}  # frame_number: expected_processing_time

        # Session info
        self.session_start_time = time.time()
        self.total_frames_processed = 0
        self.total_obstacles_detected = 0

        # Evolution matrices
        self.performance_matrix = []
        self.accuracy_matrix = []
        self.efficiency_matrix = []

    def ensure_evolution_dir(self):
        """Create evolution directory if it doesn't exist."""
        if not os.path.exists(self.evolution_dir):
            os.makedirs(self.evolution_dir)

    def record_frame_metrics(self, frame_number: int, fps: float, turning_angle: float,
                           processing_time: float, obstacles_detected: bool = False):
        """Record metrics for a single frame."""
        current_time = time.time()

        self.fps_data.append(fps)
        self.angle_data.append(turning_angle)
        self.processing_times.append(processing_time)
        self.frame_timestamps.append(current_time)
        self.obstacle_detection_data.append(obstacles_detected)

        self.total_frames_processed += 1
        if obstacles_detected:
            self.total_obstacles_detected += 1

    def set_ground_truth(self, ground_truth_file: str = None):
        """Load or set ground truth data for accuracy calculations."""
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r') as f:
                data = json.load(f)
                self.ground_truth_angles = data.get('angles', {})
                self.ground_truth_timeframes = data.get('timeframes', {})
        else:
            # Generate synthetic ground truth for demonstration
            for i in range(len(self.angle_data)):
                # Simulate ideal angles (straight with small variations)
                ideal_angle = np.random.normal(0, 5)  # Mean 0, std 5 degrees
                self.ground_truth_angles[i] = ideal_angle

                # Simulate ideal processing times
                ideal_time = 0.033  # ~30 FPS ideal
                self.ground_truth_timeframes[i] = ideal_time

    def calculate_angle_accuracy(self) -> Dict[str, float]:
        """Calculate angle prediction accuracy metrics."""
        if not self.ground_truth_angles:
            self.set_ground_truth()

        angle_errors = []
        for i, predicted_angle in enumerate(self.angle_data):
            if i in self.ground_truth_angles:
                error = abs(predicted_angle - self.ground_truth_angles[i])
                angle_errors.append(error)

        if not angle_errors:
            return {"mean_error": 0, "std_error": 0, "max_error": 0, "accuracy_percentage": 0}

        mean_error = np.mean(angle_errors)
        std_error = np.std(angle_errors)
        max_error = np.max(angle_errors)

        # Calculate accuracy as percentage (within 10 degrees considered accurate)
        accurate_predictions = sum(1 for error in angle_errors if error <= 10)
        accuracy_percentage = (accurate_predictions / len(angle_errors)) * 100

        return {
            "mean_error": mean_error,
            "std_error": std_error,
            "max_error": max_error,
            "accuracy_percentage": accuracy_percentage,
            "total_predictions": len(angle_errors)
        }

    def calculate_timeframe_accuracy(self) -> Dict[str, float]:
        """Calculate processing time accuracy and efficiency metrics."""
        if not self.ground_truth_timeframes:
            self.set_ground_truth()

        time_deviations = []
        for i, actual_time in enumerate(self.processing_times):
            if i in self.ground_truth_timeframes:
                deviation = abs(actual_time - self.ground_truth_timeframes[i])
                time_deviations.append(deviation)

        if not time_deviations:
            return {"mean_deviation": 0, "efficiency_score": 0, "real_time_capability": 0}

        mean_deviation = np.mean(time_deviations)

        # Efficiency score (0-100, higher is better)
        efficiency_score = max(0, 100 - (mean_deviation * 1000))  # Scale deviation

        # Real-time capability (percentage of frames processed within real-time constraints)
        real_time_frames = sum(1 for t in self.processing_times if t <= 0.033)  # 30 FPS
        real_time_capability = (real_time_frames / len(self.processing_times)) * 100

        return {
            "mean_deviation": mean_deviation,
            "efficiency_score": efficiency_score,
            "real_time_capability": real_time_capability,
            "average_processing_time": np.mean(self.processing_times)
        }

    def calculate_fps_metrics(self) -> Dict[str, float]:
        """Calculate FPS-related metrics."""
        if not self.fps_data:
            return {"average_fps": 0, "min_fps": 0, "max_fps": 0, "fps_stability": 0}

        average_fps = np.mean(self.fps_data)
        min_fps = np.min(self.fps_data)
        max_fps = np.max(self.fps_data)
        fps_std = np.std(self.fps_data)

        # FPS stability score (0-100, higher is better)
        fps_stability = max(0, 100 - (fps_std / average_fps * 100))

        return {
            "average_fps": average_fps,
            "min_fps": min_fps,
            "max_fps": max_fps,
            "fps_stability": fps_stability,
            "fps_std": fps_std
        }

    def generate_evolution_matrices(self):
        """Generate evolution matrices for performance analysis."""
        # Performance matrix: [FPS, Processing_Time, Memory_Usage, Accuracy]
        fps_metrics = self.calculate_fps_metrics()
        angle_metrics = self.calculate_angle_accuracy()
        time_metrics = self.calculate_timeframe_accuracy()

        # Normalize metrics to 0-100 scale for matrix representation
        performance_row = [
            min(fps_metrics["average_fps"] / 60 * 100, 100),  # Normalize to 60 FPS max
            min(time_metrics["efficiency_score"], 100),
            min(angle_metrics["accuracy_percentage"], 100),
            min(time_metrics["real_time_capability"], 100)
        ]

        self.performance_matrix.append(performance_row)

        # Accuracy matrix: [Angle_Accuracy, Time_Accuracy, Detection_Rate]
        detection_rate = (self.total_obstacles_detected / max(self.total_frames_processed, 1)) * 100
        accuracy_row = [
            angle_metrics["accuracy_percentage"],
            time_metrics["real_time_capability"],
            min(detection_rate, 100)
        ]

        self.accuracy_matrix.append(accuracy_row)

        # Efficiency matrix: [Resource_Usage, Speed, Stability]
        efficiency_row = [
            time_metrics["efficiency_score"],
            min(fps_metrics["average_fps"] / 30 * 100, 100),  # Normalize to 30 FPS
            fps_metrics["fps_stability"]
        ]

        self.efficiency_matrix.append(efficiency_row)

    def save_evolution_plots(self, report_path: str):
        """Generate and save evolution plots."""
        if HAS_SEABORN:
            plt.style.use('seaborn-v0_8')
        else:
            plt.style.use('default')

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Road Navigation System Evolution Analysis', fontsize=16, fontweight='bold')

        # FPS over time
        axes[0, 0].plot(self.fps_data, color='blue', linewidth=2)
        axes[0, 0].set_title('FPS Performance Over Time')
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True, alpha=0.3)

        # Angle distribution
        axes[0, 1].hist(self.angle_data, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Turning Angle Distribution')
        axes[0, 1].set_xlabel('Angle (degrees)')
        axes[0, 1].set_ylabel('Frequency')

        # Processing time over time
        axes[0, 2].plot(self.processing_times, color='red', linewidth=2)
        axes[0, 2].set_title('Processing Time Over Time')
        axes[0, 2].set_xlabel('Frame Number')
        axes[0, 2].set_ylabel('Processing Time (s)')
        axes[0, 2].grid(True, alpha=0.3)

        # Performance matrix heatmap
        if self.performance_matrix:
            matrix_data = np.array(self.performance_matrix).T
            if HAS_SEABORN:
                sns.heatmap(matrix_data,
                           xticklabels=[f'Session {i+1}' for i in range(len(self.performance_matrix))],
                           yticklabels=['FPS Score', 'Efficiency', 'Angle Accuracy', 'Real-time'],
                           annot=True, fmt='.1f', cmap='RdYlGn', ax=axes[1, 0])
            else:
                im = axes[1, 0].imshow(matrix_data, cmap='RdYlGn', aspect='auto')
                axes[1, 0].set_xticks(range(len(self.performance_matrix)))
                axes[1, 0].set_xticklabels([f'Session {i+1}' for i in range(len(self.performance_matrix))])
                axes[1, 0].set_yticks(range(4))
                axes[1, 0].set_yticklabels(['FPS Score', 'Efficiency', 'Angle Accuracy', 'Real-time'])
                plt.colorbar(im, ax=axes[1, 0])
            axes[1, 0].set_title('Performance Evolution Matrix')

        # Accuracy trends
        if len(self.angle_data) > 10:
            window_size = len(self.angle_data) // 10
            rolling_accuracy = []
            for i in range(0, len(self.angle_data) - window_size, window_size):
                window_angles = self.angle_data[i:i+window_size]
                accuracy = sum(1 for angle in window_angles if abs(angle) <= 10) / len(window_angles) * 100
                rolling_accuracy.append(accuracy)

            axes[1, 1].plot(rolling_accuracy, marker='o', linewidth=2, markersize=6)
            axes[1, 1].set_title('Rolling Accuracy Trend')
            axes[1, 1].set_xlabel('Time Window')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].grid(True, alpha=0.3)

        # System efficiency radar chart
        if self.efficiency_matrix:
            angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            values = self.efficiency_matrix[-1] + [self.efficiency_matrix[-1][0]]  # Last session + close circle

            axes[1, 2].remove()
            ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
            ax_radar.plot(angles, values, 'o-', linewidth=2, color='purple')
            ax_radar.fill(angles, values, alpha=0.25, color='purple')
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(['Resource Usage', 'Speed', 'Stability'])
            ax_radar.set_title('System Efficiency Radar', pad=20)

        plt.tight_layout()
        plot_path = report_path.replace('.txt', '_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_path

    def generate_report(self) -> str:
        """Generate comprehensive evolution report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report-{timestamp}.txt"
        report_path = os.path.join(self.evolution_dir, report_filename)

        # Calculate all metrics
        fps_metrics = self.calculate_fps_metrics()
        angle_metrics = self.calculate_angle_accuracy()
        time_metrics = self.calculate_timeframe_accuracy()

        # Generate evolution matrices
        self.generate_evolution_matrices()

        # Create report content
        session_duration = time.time() - self.session_start_time

        report_content = f"""
========================================
ROAD NAVIGATION SYSTEM EVOLUTION REPORT
========================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session Duration: {session_duration:.2f} seconds
Total Frames Processed: {self.total_frames_processed}
Total Obstacles Detected: {self.total_obstacles_detected}

========================================
PERFORMANCE METRICS
========================================

FPS ANALYSIS:
- Average FPS: {fps_metrics['average_fps']:.2f}
- Minimum FPS: {fps_metrics['min_fps']:.2f}
- Maximum FPS: {fps_metrics['max_fps']:.2f}
- FPS Stability Score: {fps_metrics['fps_stability']:.2f}%
- FPS Standard Deviation: {fps_metrics['fps_std']:.2f}

ANGLE ACCURACY:
- Mean Angle Error: {angle_metrics['mean_error']:.2f} degrees
- Angle Error Std Dev: {angle_metrics['std_error']:.2f} degrees
- Maximum Angle Error: {angle_metrics['max_error']:.2f} degrees
- Angle Accuracy Percentage: {angle_metrics['accuracy_percentage']:.2f}%
- Total Angle Predictions: {angle_metrics['total_predictions']}

TIMEFRAME ACCURACY:
- Mean Processing Time Deviation: {time_metrics['mean_deviation']:.4f} seconds
- Processing Efficiency Score: {time_metrics['efficiency_score']:.2f}%
- Real-time Capability: {time_metrics['real_time_capability']:.2f}%
- Average Processing Time: {time_metrics['average_processing_time']:.4f} seconds

========================================
EVOLUTION MATRICES
========================================

PERFORMANCE MATRIX (0-100 scale):
{self._format_matrix(self.performance_matrix, ['FPS Score', 'Efficiency', 'Angle Accuracy', 'Real-time'])}

ACCURACY MATRIX (0-100 scale):
{self._format_matrix(self.accuracy_matrix, ['Angle Accuracy', 'Time Accuracy', 'Detection Rate'])}

EFFICIENCY MATRIX (0-100 scale):
{self._format_matrix(self.efficiency_matrix, ['Resource Usage', 'Speed', 'Stability'])}

========================================
SYSTEM ANALYSIS
========================================

STRENGTHS:
{self._analyze_strengths(fps_metrics, angle_metrics, time_metrics)}

AREAS FOR IMPROVEMENT:
{self._analyze_weaknesses(fps_metrics, angle_metrics, time_metrics)}

RECOMMENDATIONS:
{self._generate_recommendations(fps_metrics, angle_metrics, time_metrics)}

========================================
STATISTICAL SUMMARY
========================================

Frame Processing Statistics:
- Total frames: {len(self.fps_data)}
- Frames with obstacles: {sum(self.obstacle_detection_data)}
- Obstacle detection rate: {(sum(self.obstacle_detection_data) / max(len(self.fps_data), 1)) * 100:.2f}%

Performance Distribution:
- Frames above 30 FPS: {sum(1 for fps in self.fps_data if fps >= 30)}
- Frames above 25 FPS: {sum(1 for fps in self.fps_data if fps >= 25)}
- Frames below 15 FPS: {sum(1 for fps in self.fps_data if fps < 15)}

Angle Distribution:
- Straight ahead (±5°): {sum(1 for angle in self.angle_data if abs(angle) <= 5)}
- Moderate turns (5-20°): {sum(1 for angle in self.angle_data if 5 < abs(angle) <= 20)}
- Sharp turns (>20°): {sum(1 for angle in self.angle_data if abs(angle) > 20)}

========================================
END OF REPORT
========================================
"""

        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Generate and save plots
        plot_path = self.save_evolution_plots(report_path)

        print(f"Evolution report generated: {report_path}")
        print(f"Evolution plots generated: {plot_path}")

        return report_path

    def _format_matrix(self, matrix, labels):
        """Format matrix for display in report."""
        if not matrix:
            return "No data available"

        formatted = "┌" + "─" * 60 + "┐\n"
        for i, label in enumerate(labels):
            values = [row[i] if i < len(row) else 0 for row in matrix]
            avg_value = np.mean(values) if values else 0
            formatted += f"│ {label:<20} │ {avg_value:>8.2f} │\n"
        formatted += "└" + "─" * 60 + "┘"
        return formatted

    def _analyze_strengths(self, fps_metrics, angle_metrics, time_metrics):
        """Analyze system strengths."""
        strengths = []

        if fps_metrics['average_fps'] > 25:
            strengths.append("- High average FPS performance (>25 FPS)")
        if fps_metrics['fps_stability'] > 80:
            strengths.append("- Stable FPS performance")
        if angle_metrics['accuracy_percentage'] > 80:
            strengths.append("- High angle prediction accuracy")
        if time_metrics['real_time_capability'] > 90:
            strengths.append("- Excellent real-time processing capability")

        return '\n'.join(strengths) if strengths else "- System performance needs improvement across all metrics"

    def _analyze_weaknesses(self, fps_metrics, angle_metrics, time_metrics):
        """Analyze system weaknesses."""
        weaknesses = []

        if fps_metrics['average_fps'] < 20:
            weaknesses.append("- Low average FPS performance (<20 FPS)")
        if fps_metrics['fps_stability'] < 60:
            weaknesses.append("- Unstable FPS performance")
        if angle_metrics['accuracy_percentage'] < 70:
            weaknesses.append("- Low angle prediction accuracy")
        if time_metrics['real_time_capability'] < 80:
            weaknesses.append("- Insufficient real-time processing capability")

        return '\n'.join(weaknesses) if weaknesses else "- No significant weaknesses detected"

    def _generate_recommendations(self, fps_metrics, angle_metrics, time_metrics):
        """Generate improvement recommendations."""
        recommendations = []

        if fps_metrics['average_fps'] < 25:
            recommendations.append("- Consider optimizing depth estimation model or reducing input resolution")
        if angle_metrics['accuracy_percentage'] < 80:
            recommendations.append("- Improve obstacle detection algorithm or calibrate angle calculation")
        if time_metrics['real_time_capability'] < 90:
            recommendations.append("- Optimize processing pipeline or consider parallel processing")
        if fps_metrics['fps_stability'] < 70:
            recommendations.append("- Implement frame buffering or dynamic quality adjustment")

        return '\n'.join(recommendations) if recommendations else "- System is performing well, continue monitoring"
