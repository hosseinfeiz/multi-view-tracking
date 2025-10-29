import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import os

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    config_file = args.config_file or str(dataset_path / "tracks" / "tracking_config.json")
    output_dir = args.output_dir or str(dataset_path / "tracks")
    
    tracking_config = TrackingConfig(
        num_persons=args.num_persons,
        num_cameras=args.num_cameras,
        detection_confidence=0.3
    )
    
    spring_config = SpringDamperConfig(
        spring_constant=args.spring_constant,
        damping_coefficient=args.damping_coefficient
    )
    
    camera_params = parse_camera_parameters(args.camera_params)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    tracker = StableTracker(tracking_config, spring_config, camera_params)
    visualizer = VisualizationEngine(tracker.projection)
    xml_output = XMLOutputManager(output_dir, camera_params)
    
    video_paths = load_video_streams(args.dataset_path, args.num_cameras)
    initial_config = get_initialization_data(config_file, video_paths, args.num_cameras)
    
    if initial_config is None:
        return
    
    track_mapping = tracker.initialize_from_config(initial_config)
    
    caps = setup_video_captures(video_paths)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    
    frame_idx = 0
    stable_tracking_started = False
    
    if args.headless:
        pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while frame_idx < total_frames:
        if frame_idx % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        frames_original, frames_tracking, scales = read_frame_batch_optimized(caps, tracker.device)
        if frames_original is None:
            break
        
        frames_np = []
        for frame in frames_original:
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            frames_np.append(frame_np)
        
        # Initialize XMem with ground truth on first frame
        if frame_idx == 0 and not stable_tracking_started:
            tracker.add_xmem_masks(frames_np, initial_config)
            stable_tracking_started = True
        
        try:
            result = tracker.process_frame(frames_tracking, scales)
            tracks = result['tracks']
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            continue
        
        # Create compatible tracks for XML output
        compatible_tracks = []
        for track in tracks:
            # Show all tracks, not just active ones
            class CompatTrack:
                def __init__(self, track_obj):
                    self.id = track_obj.id
                    self.bboxes = [None] * tracking_config.num_cameras
                    self.confidence_scores = [0.0] * tracking_config.num_cameras
                    self.occlusion_states = ['MISSING'] * tracking_config.num_cameras
                    
                    for view_idx in range(tracking_config.num_cameras):
                        # Only include bbox if track is actually visible in this view
                        if track_obj.is_visible_in_view(view_idx):
                            self.bboxes[view_idx] = track_obj.bboxes[view_idx]
                            self.confidence_scores[view_idx] = track_obj.view_confidences.get(view_idx, 0.0)
                            self.occlusion_states[view_idx] = 'VISIBLE'
                        else:
                            # Track is occluded or missing - don't show bbox
                            self.bboxes[view_idx] = None
                            self.confidence_scores[view_idx] = 0.0
                            occlusion_state = track_obj.occlusion_states.get(view_idx, 'MISSING')
                            self.occlusion_states[view_idx] = occlusion_state
            
            compatible_tracks.append(CompatTrack(track))
        
        xml_output.add_track_detections(compatible_tracks, frame_idx)
        
        # Visualization
        if not args.headless:
            try:
                mosaic, scale = visualizer.create_mosaic(frames_original)
                mosaic_with_tracks = visualizer.draw_tracks(mosaic, tracks, scale)
                final_mosaic = visualizer.add_performance_info(
                    mosaic_with_tracks, tracker.performance_monitor, frame_idx, tracks
                )
                
                # Add stability indicators
                stability_info = []
                # for track in tracks:
                #     if track.is_active():
                #         stability_info.append(f"ID{track.id}: C{track.confidence:.2f} G{track.geometric_consistency:.2f}")
                
                y_offset = final_mosaic.shape[0] - 150
                for i, info in enumerate(stability_info[:5]):
                    cv2.putText(final_mosaic, info, (15, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Multi-View Tracker", final_mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
            except Exception:
                pass
        else:
            pbar.update(1)
        
        if frame_idx % 100 == 0:
            xml_output.save_xml_files()
            torch.cuda.empty_cache()
            gc.collect()
        
        frame_idx += 1
    
    if args.headless:
        pbar.close()
    
    for cap in caps:
        cap.release()
    
    if not args.headless:
        cv2.destroyAllWindows()
    
    xml_output.save_xml_files()
    
    torch.cuda.empty_cache()
    gc.collect()

def read_frame_batch_optimized(caps, device, tracking_size=360):
    frames_original = []
    frames_tracking = []
    scales = []
    
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            return None, None, None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        with torch.no_grad():
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames_original.append(frame_tensor.to(device, non_blocking=True))
            
            if max(h, w) > tracking_size:
                scale = tracking_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_small = cv2.resize(frame, (new_w, new_h))
                frame_tensor_small = torch.from_numpy(frame_small).permute(2, 0, 1).float() / 255.0
                frames_tracking.append(frame_tensor_small.to(device, non_blocking=True))
                scales.append(1.0 / scale)
            else:
                frames_tracking.append(frame_tensor.to(device, non_blocking=True))
                scales.append(1.0)
    
    return frames_original, frames_tracking, scales

if __name__ == "__main__":
    from pose_estimation.track.utils import (
        TrackingConfig, SpringDamperConfig, parse_camera_parameters, get_initialization_data,
        load_video_streams, setup_video_captures, create_argument_parser,
        DetectionEngine, XMLOutputManager
    )
    from pose_estimation.track.visualization import VisualizationEngine
    from pose_estimation.track.tracker_core import StableTracker
    
    main()