#!/usr/bin/env python3
"""
Demo script to test the evolution tracking system for road navigation.
This script shows how to use the enhanced roadmaxAngle.py with evolution tracking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roadmaxAngle import RoadNavigationSystem

def main():
    """Main demo function."""
    print("=" * 60)
    print("ROAD NAVIGATION EVOLUTION TRACKING DEMO")
    print("=" * 60)
    print()

    # Configuration options
    video_path = "path1.mp4"  # Your video file

    # You can also use webcam by setting video_path = 0
    # video_path = 0  # For webcam input

    print(f"Video source: {video_path}")
    print()

    # Initialize navigation system with evolution tracking enabled
    print("Initializing Road Navigation System...")
    navigation_system = RoadNavigationSystem(
        model_type="MiDaS_small",           # You can try "DPT_Large" for better accuracy but slower
        input_size=(128, 128),              # Smaller for faster processing
        depth_threshold_factor=0.5,         # Adjust for obstacle sensitivity
        enable_evolution_tracking=True      # Enable evolution tracking
    )

    print("System initialized successfully!")
    print()
    print("Instructions:")
    print("- Press 'q' to quit and generate evolution report")
    print("- The system will automatically track:")
    print("  * FPS performance")
    print("  * Angle prediction accuracy")
    print("  * Processing time efficiency")
    print("  * Obstacle detection statistics")
    print()
    print("Starting video processing...")
    print("-" * 40)

    # Run the navigation system
    try:
        navigation_system.run(video_path)
    except FileNotFoundError:
        print(f"Error: Video file '{video_path}' not found!")
        print("Please make sure the video file exists or use webcam (video_path = 0)")
        return
    except Exception as e:
        print(f"Error during processing: {e}")
        return

    print()
    print("=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("Check the /evolutions/ folder for:")
    print("- Detailed text report with all metrics")
    print("- Evolution plots and visualizations")
    print("- Performance matrices")

if __name__ == "__main__":
    main()
