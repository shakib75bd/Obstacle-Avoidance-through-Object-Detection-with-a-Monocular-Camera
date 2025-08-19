#!/usr/bin/env python3
"""
Test script for evolution tracking system.
This generates synthetic data to test the evolution matrices and reporting functionality.
"""

import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evolution_tracker import EvolutionTracker

def test_evolution_tracking():
    """Test the evolution tracking system with synthetic data."""
    print("Testing Evolution Tracking System")
    print("=" * 50)

    # Create evolution tracker
    tracker = EvolutionTracker()

    # Generate synthetic test data (simulating 100 frames of processing)
    print("Generating synthetic test data...")
    np.random.seed(42)  # For reproducible results

    for frame in range(100):
        # Simulate varying performance
        base_fps = 25 + np.random.normal(0, 5)
        fps = max(10, min(60, base_fps))  # Keep FPS between 10-60

        # Simulate turning angles with some realistic patterns
        if frame < 30:
            angle = np.random.normal(0, 3)  # Mostly straight
        elif frame < 60:
            angle = np.random.normal(-15, 8)  # Left turn section
        else:
            angle = np.random.normal(10, 5)  # Right turn section

        # Simulate processing time (inversely related to FPS)
        processing_time = 1.0 / fps + np.random.normal(0, 0.001)

        # Simulate obstacle detection (more likely during turns)
        obstacles_detected = abs(angle) > 10 and np.random.random() > 0.3

        # Record metrics
        tracker.record_frame_metrics(
            frame_number=frame,
            fps=fps,
            turning_angle=angle,
            processing_time=processing_time,
            obstacles_detected=obstacles_detected
        )

        if frame % 20 == 0:
            print(f"Processed frame {frame}/100")

    print("Synthetic data generation complete!")
    print()

    # Test individual metric calculations
    print("Testing metric calculations...")
    fps_metrics = tracker.calculate_fps_metrics()
    angle_metrics = tracker.calculate_angle_accuracy()
    time_metrics = tracker.calculate_timeframe_accuracy()

    print(f"Average FPS: {fps_metrics['average_fps']:.2f}")
    print(f"Angle Accuracy: {angle_metrics['accuracy_percentage']:.2f}%")
    print(f"Real-time Capability: {time_metrics['real_time_capability']:.2f}%")
    print()

    # Generate evolution matrices
    print("Generating evolution matrices...")
    tracker.generate_evolution_matrices()

    print(f"Performance matrix entries: {len(tracker.performance_matrix)}")
    print(f"Accuracy matrix entries: {len(tracker.accuracy_matrix)}")
    print(f"Efficiency matrix entries: {len(tracker.efficiency_matrix)}")
    print()

    # Generate full report
    print("Generating comprehensive report...")
    report_path = tracker.generate_report()

    print(f"✅ Report generated successfully!")
    print(f"📄 Report location: {report_path}")

    # Check if files exist
    if os.path.exists(report_path):
        file_size = os.path.getsize(report_path)
        print(f"📊 Report size: {file_size} bytes")

        plot_path = report_path.replace('.txt', '_plots.png')
        if os.path.exists(plot_path):
            plot_size = os.path.getsize(plot_path)
            print(f"📈 Plots size: {plot_size} bytes")
        else:
            print("⚠️  Plot file not found (may be due to missing matplotlib/seaborn)")

    print()
    print("Evolution tracking test completed successfully! 🎉")
    return report_path

if __name__ == "__main__":
    try:
        test_evolution_tracking()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
