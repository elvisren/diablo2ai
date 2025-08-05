import sys
import time
import cv2
from detector import YoloTracker
from video_processor import VideoProcessor
from visualizer import Visualizer
from fps_counter import FPSCounter


def main():
    # Parse command line arguments
    source = 0  # Default camera
    model_path = None
    is_video = False

    if len(sys.argv) > 1:
        source_arg = sys.argv[1]
        try:
            # Try as integer camera index first
            source = int(source_arg)
        except ValueError:
            # Treat as video file
            source = source_arg
            is_video = True

    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    # Initialize components
    try:
        video_source = VideoProcessor(source, is_video)
        print(f"Opened video source: {video_source.get_source_info()}")
    except Exception as e:
        print(f"Error opening source: {str(e)}")
        return

    tracker = None
    if model_path:
        try:
            # Initialize tracker with custom configuration
            tracker = YoloTracker(
                model_path=model_path,
                conf_threshold=0.5,
                iou_threshold=0.5,
                img_size=640,
                tracker='botsort.yaml',  # or 'bytetrack.yaml'
                device='auto',
                verbose=True
            )
        except Exception as e:
            print(f"Error loading tracker: {str(e)}")
            tracker = None

    visualizer = Visualizer()
    fps_counter = FPSCounter()

    # Main processing loop
    window_name = "YOLO Tracking" + (" - Video" if is_video else " - Camera")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()
    tracks = []
    tracking_time = 0

    print("\nControls:")
    print("- Press 'ESC' to exit")
    print("- Press 'SPACE' to pause/resume video")
    print("- Press '+' to increase processing speed")
    print("- Press '-' to decrease processing speed")

    paused = False
    video_speed = 1.0  # Playback speed multiplier

    try:
        while True:
            if not paused:
                # Timing for processing measurement
                process_start = time.time()

                # Get next frame (with timing control)
                frame = video_source.get_frame()
                if frame is None:
                    break

                # Run object tracking
                if tracker:
                    # Convert to RGB for tracker
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tracks, tracking_time = tracker.track(rgb_frame, persist=True)
                    visualizer.draw_tracks(frame, tracks)

                # Update performance metrics
                current_time = time.time()
                frame_count += 1
                fps_counter.update()
                fps = fps_counter.get_fps()
                proc_time = (current_time - process_start) * 1000
                latency = video_source.latency * 1000

                # Draw performance overlay
                visualizer.draw_performance(
                    frame, fps, latency, proc_time,
                    video_source.get_source_info(),
                    tracking_time * 1000,
                    len(tracks)
                )

                # Display frame
                cv2.imshow(window_name, frame)

            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == 32:  # SPACE key
                paused = not paused
            elif key == ord('+'):
                video_speed = min(video_speed * 1.2, 5.0)
                if is_video:
                    print(f"Speed increased: {video_speed:.1f}x")
            elif key == ord('-'):
                video_speed = max(video_speed * 0.8, 0.2)
                if is_video:
                    print(f"Speed decreased: {video_speed:.1f}x")

    finally:
        # Cleanup and final report
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print("\n===== Performance Summary =====")
        print(f"Frames processed: {frame_count}")
        print(f"Frames dropped: {video_source.dropped_frames}")
        print(f"Avg FPS: {avg_fps:.2f}")
        if tracker:
            print(f"Avg tracking time: {tracking_time * 1000:.1f} ms")
        print(f"Avg processing time: {proc_time:.1f} ms")
        print(f"Avg latency: {latency:.1f} ms")

        video_source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()