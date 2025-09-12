import os
import sys
import torch
import json
import time
import warnings
import numpy as np
from os import path
from pathlib import Path
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace

# Add DEVA to path
sys.path.append('/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA')

# Import DEVA components
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.automatic_sam import get_sam_model
from deva.ext.automatic_processor import process_frame_automatic as process_frame

# Modified SimpleVideoReader that supports both PNG and JPG files
class ModifiedSimpleVideoReader(SimpleVideoReader):
    """
    This is a modified version of SimpleVideoReader that supports both PNG and JPG files
    """
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # Look for both PNG and JPG/JPEG files
        self.frames = sorted([frame for frame in os.listdir(self.image_dir) 
                            if frame.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"ModifiedSimpleVideoReader found {len(self.frames)} images")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch\.cuda\.amp\.autocast.*")
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'  # Use a temporary directory for matplotlib config

class DEVATracker:
    """
    A wrapper class for DEVA tracking functionality.
    """
    
    def __init__(self, gpu_id=0):
        """
        Initialize the DEVA tracker.
        
        Args:
            gpu_id (int): GPU ID to use for processing.
        """
        # Set default paths
        self.deva_model_path = '/nas/project_data/B1_Behavior/rush/kaan/first_play_local/models/deva/saves/DEVA-propagation.pth'
        self.sam_model_path = '/nas/project_data/B1_Behavior/rush/kaan/first_play_local/models/deva/saves/sam_vit_h_4b8939.pth'
        self.mobile_sam_path = '/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA/saves/mobile_sam.pt'
        
        # Set CUDA device
        print(f"Step 2: Setting CUDA device: {gpu_id}")
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.empty_cache()
        print(f"CUDA device set, available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
        
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if torch.cuda.is_available():
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available")
            
        # Disable gradient computation
        torch.autograd.set_grad_enabled(False)
        
    def run_tracking(self, 
                     img_path, 
                     output_path, 
                     chunk_size=8, 
                     temporal_setting="semionline", 
                     size=-1, 
                     mem_every=5, 
                     detection_every=10, 
                     max_num_objects=100, 
                     sam_variant="original",
                     amp=True,
                     num_voting_frames=5,
                     save_all=True):
        """
        Run DEVA tracking on images in the specified path.
        
        Args:
            img_path (str): Path to the input images.
            output_path (str): Path to save the output.
            chunk_size (int): The chunk size for processing.
            temporal_setting (str): Temporal setting to use, e.g., 'semionline' or 'online'.
            size (int): The size parameter for image resolution (-1 for original size).
            mem_every (int): Memory savings frequency.
            detection_every (int): Detection frequency.
            max_num_objects (int): Maximum number of objects to track.
            sam_variant (str): SAM variant to use ('original', 'mobile', or 'fast').
            amp (bool): Whether to use automatic mixed precision.
            num_voting_frames (int): Number of voting frames.
            save_all (bool): Whether to save all intermediate results.
            
        Returns:
            bool: True if tracking was successful, False otherwise.
        """
        print("\n=== Starting DEVATracker.run_tracking ===")
        print(f"Input path: {img_path}")
        print(f"Output path: {output_path}")
        print(f"Parameters: chunk_size={chunk_size}, temporal_setting={temporal_setting}, size={size}, mem_every={mem_every}, detection_every={detection_every}, sam_variant={sam_variant}")
        
        try:
            print("Step 1: Importing required modules...")
            import os
            import sys
            import torch
            
            # Add DEVA project path to system path
            deva_path = '/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA'
            if deva_path not in sys.path:
                sys.path.append(deva_path)
                print(f"Added {deva_path} to sys.path")
            
            from deva.inference.inference_core import DEVAInferenceCore
            from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
            from deva.inference.result_utils import ResultSaver
            from deva.inference.demo_utils import flush_buffer
            from deva.ext.automatic_processor import process_frame_automatic as process_frame
            from torch.utils.data import DataLoader
            import time
            from os import path
            print("All modules imported successfully")
            
            # Some sanity checks
            if not path.isdir(img_path):
                print(f"Error: img_path {img_path} is not a valid directory.")
                return False
                
            print(f"Verified input directory exists: {img_path}")
            
            # Prepare argument object
            print("Step 3: Preparing arguments...")
            # Set model path to default location if none provided
            deva_model_path = self.deva_model_path
            args = Namespace(
                model=deva_model_path,
                output=output_path,
                save_all=save_all,
                amp=amp,
                chunk_size=chunk_size,
                temporal_setting=temporal_setting,
                mem_every=mem_every,
                detection_every=detection_every,
                max_num_objects=max_num_objects,
                size=size,
                sam_variant=sam_variant,
                num_voting_frames=num_voting_frames,
                # Required configuration parameters from the demo config
                key_dim=64,
                value_dim=512,
                pix_feat_dim=512,
                disable_long_term=False,
                max_mid_term_frames=10,
                min_mid_term_frames=5,
                max_long_term_elements=10000,
                num_prototypes=128,
                top_k=30,
                max_missed_detection_count=5,
                suppress_small_objects=False,  # Required by automatic_processor
                # SAM parameters
                SAM_PRED_IOU_THRESHOLD=0.88,
                SAM_OVERLAP_THRESHOLD=0.8,
                SAM_NUM_POINTS_PER_SIDE=16,
                SAM_NUM_POINTS_PER_BATCH=16
            )
            print("Arguments prepared successfully")
            
            # Get model and config
            print("Step 4: Loading DEVA model and configuration...")
            deva_model, cfg, args = self._get_model_and_config(args)
            print("DEVA model loaded successfully")
            
            # Get SAM model (using the get_sam_model to ensure properly setup)
            try:
                # We need to make sure all required SAM parameters are in the config
                required_sam_params = {
                    'SAM_ENCODER_VERSION': 'vit_h',
                    'SAM_CHECKPOINT_PATH': '/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth',
                    'MOBILE_SAM_CHECKPOINT_PATH': '/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA/saves/mobile_sam.pt',
                    'SAM_NUM_POINTS_PER_SIDE': 32,
                    'SAM_NUM_POINTS_PER_BATCH': 64,
                    'SAM_PRED_IOU_THRESHOLD': 0.88,
                    'SAM_OVERLAP_THRESHOLD': 0.7
                }
                
                # Add any missing parameters to config
                for key, value in required_sam_params.items():
                    if key not in cfg:
                        cfg[key] = value
                        print(f"Added missing SAM parameter: {key}={value}")
                
                from deva.ext.automatic_sam import get_sam_model
                sam_model = get_sam_model(cfg, 'cuda')
                print(f"Successfully loaded SAM model variant: {args.sam_variant}")
            except Exception as e:
                print(f"Error loading SAM model: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Prepare output directory
            print("Step 5: Preparing output directory...")
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)
            print(f"Output directory created/verified: {out_path}")
            
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Verify temporal setting
            cfg['temporal_setting'] = args.temporal_setting.lower()
            assert cfg['temporal_setting'] in ['semionline', 'online']
            
            # Initialize dataset with modified reader that supports JPG files
            print("Step 6: Initializing video reader and data loader...")
            reader = ModifiedSimpleVideoReader(img_path)
            vid_length = len(reader)
            print(f"Found {vid_length} images in input directory")
            
            # No batch processing, just iterate through all frames
            loader = DataLoader(reader, batch_size=None, shuffle=False, num_workers=2, collate_fn=no_collate)
            print("DataLoader initialized")
            
            # Enable long-term memory if specified
            if not hasattr(args, 'disable_long_term'):
                cfg['enable_long_term'] = True
            else:
                cfg['enable_long_term'] = not args.disable_long_term
                
            # Configure long-term settings
            cfg['enable_long_term_count_usage'] = (
                cfg['enable_long_term'] and 
                ((cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) * 
                 cfg['num_prototypes'] >= cfg['max_long_term_elements']))
                
            # Add all required configuration parameters with defaults matching the user's config
            # Core settings
            cfg['suppress_small_objects'] = args.suppress_small_objects if hasattr(args, 'suppress_small_objects') else False
            cfg['max_missed_detection_count'] = args.max_missed_detection_count if hasattr(args, 'max_missed_detection_count') else 5
            cfg['max_num_objects'] = args.max_num_objects if hasattr(args, 'max_num_objects') else 20
            
            # Model dimensions
            if 'key_dim' not in cfg:
                cfg['key_dim'] = 64
            if 'value_dim' not in cfg:
                cfg['value_dim'] = 512
            if 'pix_feat_dim' not in cfg:
                cfg['pix_feat_dim'] = 512
                
            # Memory settings
            if 'max_mid_term_frames' not in cfg:
                cfg['max_mid_term_frames'] = 10
            if 'min_mid_term_frames' not in cfg:
                cfg['min_mid_term_frames'] = 5
            if 'max_long_term_elements' not in cfg:
                cfg['max_long_term_elements'] = 10000
            if 'num_prototypes' not in cfg:
                cfg['num_prototypes'] = 128
            if 'top_k' not in cfg:
                cfg['top_k'] = 30
                
            # SAM settings
            if 'SAM_NUM_POINTS_PER_SIDE' not in cfg:
                cfg['SAM_NUM_POINTS_PER_SIDE'] = 16
            if 'SAM_NUM_POINTS_PER_BATCH' not in cfg:
                cfg['SAM_NUM_POINTS_PER_BATCH'] = 16
            if 'SAM_PRED_IOU_THRESHOLD' not in cfg:
                cfg['SAM_PRED_IOU_THRESHOLD'] = 0.88
            if 'SAM_OVERLAP_THRESHOLD' not in cfg:
                cfg['SAM_OVERLAP_THRESHOLD'] = 0.8
                
            # Debugging info
            print(f"Configuration initialized with max_num_objects={cfg.get('max_num_objects', 'not set')}")
            
            # Initialize DEVA
            print("Step 8: Initializing DEVA inference core...")
            deva = DEVAInferenceCore(deva_model, config=cfg)
            print("DEVA inference core initialized successfully")
            deva.next_voting_frame = args.num_voting_frames - 1
            deva.enabled_long_id()
            
            # Prepare result saver - must be after DEVA initialization
            print("Step 9: Initializing result saver...")
            result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)
            print("Result saver initialized")
            
            torch.cuda.empty_cache()
            print(f"Starting tracking on {img_path}")
            tick_all = time.time()
            try:
                # Process frames with AMP if enabled
                print("Step 10: Processing frames...")
                print("Starting frame processing loop (this may take some time)")
                
                with torch.cuda.amp.autocast(enabled=args.amp):
                    for ti, (frame, im_path) in enumerate(loader):
                        print(f"Processing frame {ti+1}/{vid_length}")
                        print(f"Frame shape: {frame.shape}")
                        print(f"Frame path: {im_path}")
                        try:
                            # Process frame using the DEVA processor
                            process_frame(deva, sam_model, im_path, result_saver, ti, image_np=frame)
                            print(f"Frame {ti+1} processed successfully")
                        except Exception as frame_error:
                            print(f"Error processing frame {ti+1}: {str(frame_error)}")
                            import traceback
                            traceback.print_exc()
                            # Continue with next frame
                    
                    # Flush buffer to save remaining results
                    print("Flushing buffer to save remaining results")
                    flush_buffer(deva, result_saver)
                    print("Buffer flushed successfully")
            except Exception as e:
                print(f"Error in frame processing loop: {str(e)}")
                import traceback
                traceback.print_exc()
                print("Proceeding despite errors in frame processing...")
                # Continue execution even if tracking fails
                
            tack_all = time.time()
            print(f"Step 10: Tracking completed in {tack_all - tick_all:.2f} seconds")
            
            # Save results to JSON
            print("Step 11: Saving final results to JSON...")
            try:
                out_json = path.join(out_path, 'predictions.json')
                with open(out_json, 'w') as f:
                    json.dump(result_saver.video_json, f, indent=4)  # Use video_json attribute
                print(f"Results saved to {out_json}")
            except Exception as json_error:
                print(f"Error saving results to JSON: {str(json_error)}")
                import traceback
                traceback.print_exc()
                
            # Clear GPU cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Verify output was created
            if path.exists(path.join(out_path, 'pred.json')):
                print(f"Successfully created pred.json in {out_path}")
                return True
            else:
                print(f"WARNING: pred.json was not created in {out_path}")
                return False
                
        except Exception as e:
            print(f"Error in DEVA tracking: {str(e)}")
            import traceback
            traceback.print_exc()
            print("=== DEVATracker.run_tracking failed ===\n")
            return False
            
    def _prepare_args(self, img_path, output_path, chunk_size, temporal_setting, size, 
                     mem_every, detection_every, max_num_objects, sam_variant, save_all, 
                     amp, num_voting_frames):
        """
        Prepare arguments for DEVA tracking.
        """
        # Create a class to simulate argparse namespace
        class Args:
            pass
            
        args = Args()
        
        # Set basic attributes
        args.img_path = img_path
        args.output = output_path
        args.model = "DEVA-propagation.pth"  # Default DEVA model
        args.model_type = "deva"
        args.chunk_size = chunk_size
        args.temporal_setting = temporal_setting
        args.size = size
        
        # Model parameters needed by DEVA
        args.hidden_dim = 64
        args.deep_update_prob = 0.2
        args.no_background = False 
        args.shrink_kernel_size = 0
        args.model_path = None
        args.enable_xmem = False
        args.amp = amp
        args.mem_every = mem_every
        args.detection_every = detection_every
        args.max_num_objects = max_num_objects
        args.save_all = save_all
        args.save_scores = False
        args.num_voting_frames = num_voting_frames
        args.sam_variant = sam_variant
        
        # Core default parameters needed by the DEVA tracking system
        args.backend = "gzip"
        args.dataset = "demo"  # This is for ResultSaver
        args.split = None  # Not needed for demo mode
        args.output_all = False  # Let save_all control this
        args.benchmark = ""
        
        # SAM model parameters (needed by automatic_sam.py)
        args.SAM_ENCODER_VERSION = "vit_h"
        args.SAM_CHECKPOINT_PATH = "/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"
        args.MOBILE_SAM_CHECKPOINT_PATH = "/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA/saves/mobile_sam.pt"
        args.SAM_NUM_POINTS_PER_SIDE = 32
        args.SAM_NUM_POINTS_PER_BATCH = 64
        args.SAM_PRED_IOU_THRESHOLD = 0.88
        args.SAM_OVERLAP_THRESHOLD = 0.7
        
        # DEVA default tracking parameters
        args.enable_long_term = True
        args.num_prototypes = 128
        args.max_mid_term_frames = 10
        args.min_mid_term_frames = 5
        args.max_long_term_elements = 10000
        args.mem_every = mem_every
        args.top_k = 30
        args.stagger_updates = True
        args.disable_background = False
        
        # Additional parameters needed
        args.key_dim = 64
        args.value_dim = 512
        args.pix_feat_dim = 512
        args.use_long_term = True
        
        return args
        
    def _get_model_and_config(self, args):
        """
        Get model and config for DEVA tracking.
        """
        # Convert args to dictionary for DEVA model
        import torch
        from deva.model.network import DEVA
        
        # Convert Args object to dictionary
        cfg = {}
        for key, value in vars(args).items():
            cfg[key] = value
            
        # Set additional required parameters for network.py
        if 'key_dim' not in cfg:
            cfg['key_dim'] = 64
        if 'value_dim' not in cfg:
            cfg['value_dim'] = 512
        if 'pix_feat_dim' not in cfg:
            cfg['pix_feat_dim'] = 512
        
        print("Initializing DEVA model with config:")
        for key in ['key_dim', 'value_dim', 'pix_feat_dim', 'hidden_dim']:
            if key in cfg:
                print(f"  {key}: {cfg[key]}")
        
        # Initialize model
        try:
            deva_model = DEVA(cfg).to('cuda')
        except Exception as e:
            print(f"Error initializing DEVA model: {str(e)}")
            raise
            
        # Default model path
        base_path = '/nas/project_data/B1_Behavior/rush/kaan/old_method/deva_process/deva/Tracking-Anything-with-DEVA/saves'
        if path.isfile(args.model):
            # User provided a direct path to the model
            model_path = args.model
        elif args.model.endswith('.pth'):
            # Model name with .pth extension
            model_path = path.join(base_path, args.model)
        else:
            # Model name without .pth extension
            model_path = path.join(base_path, args.model + '.pth')
            
        print(f"Looking for model at: {model_path}")
        
        # Check if model exists
        if not path.isfile(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"Loading DEVA model from: {model_path}")
        # Load weights
        try:
            weights = torch.load(model_path)
            deva_model.load_weights(weights)
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            raise
            
        return deva_model, cfg, args
        
def process_videos(frames_dir, output_dir, gpu_id=0, num_parts=1, process_part=0, 
                  session=None, camera_view=None, chunk_size=8, temporal_setting="semionline", 
                  size=-1, mem_every=5, detection_every=10, max_num_objects=100):
    """
    Process videos with DEVA tracking with support for sessions and camera views.
    
    Args:
        frames_dir (str or Path): Directory containing input frames
        output_dir (str or Path): Base directory for output
        gpu_id (int): GPU ID to use
        num_parts (int): Number of parts to split the videos into
        process_part (int): Which part to process (0-indexed)
        session (str): Specific session to process (None for all)
        camera_view (str): Specific camera view to process (None for all)
        chunk_size (int): Chunk size for DEVA tracking
        temporal_setting (str): Temporal setting for DEVA
        size (int): Size parameter for DEVA
        mem_every (int): Memory savings frequency for DEVA
        detection_every (int): Detection frequency for DEVA
        max_num_objects (int): Maximum number of objects to track
    """
    # Convert to Path objects
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    # # Initialize DEVA tracker
    # tracker = DEVATracker(gpu_id=gpu_id)
    # ###TEST

    # success = tracker.run_tracking(
    #     img_path=str(frames_dir),
    #     output_path=str(output_dir),
    #     chunk_size=chunk_size,
    #     temporal_setting=temporal_setting,
    #     size=size,
    #     mem_every=mem_every,
    #     detection_every=detection_every,
    #     max_num_objects=max_num_objects
    # )
    # raise ValueError("TESTING ONLY")

    
    # If specific session is provided
    if session is not None:
        session_dir = frames_dir / session
        
        # Process specific session with or without specific camera view
        if camera_view is not None:
            # Process specific camera view in the session
            camera_dir = session_dir / camera_view
            if camera_dir.exists() and camera_dir.is_dir():
                process_camera_view(tracker, camera_dir, output_dir, session, camera_view, chunk_size, 
                                   temporal_setting, size, mem_every, detection_every, max_num_objects)
        else:
            # Process all camera views in the session
            camera_views = [d for d in session_dir.iterdir() if d.is_dir()]
            for camera_dir in camera_views:
                camera_view_name = camera_dir.name
                process_camera_view(tracker, camera_dir, output_dir, session, camera_view_name, chunk_size, 
                                   temporal_setting, size, mem_every, detection_every, max_num_objects)
    else:
        # Process all sessions
        sessions = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
        
        # Split into parts if needed
        if num_parts > 1:
            sessions_parts = np.array_split(sessions, num_parts)
            sessions_to_process = sessions_parts[process_part]
        else:
            sessions_to_process = sessions
            
        for session_dir in sessions_to_process:
            session_name = session_dir.name
            camera_views = [d for d in session_dir.iterdir() if d.is_dir()]
            
            for camera_dir in camera_views:
                camera_view_name = camera_dir.name
                process_camera_view(tracker, camera_dir, output_dir, session_name, camera_view_name, chunk_size, 
                                   temporal_setting, size, mem_every, detection_every, max_num_objects)

def process_camera_view(tracker, camera_dir, base_output_dir, session_name, camera_view_name, chunk_size, 
                      temporal_setting, size, mem_every, detection_every, max_num_objects):
    """
    Process a single camera view with DEVA tracking
    
    Args:
        tracker (DEVATracker): DEVA tracker instance
        camera_dir (Path): Directory containing camera view frames
        base_output_dir (Path): Base output directory
        session_name (str): Session name
        camera_view_name (str): Camera view name
        chunk_size (int): Chunk size for DEVA tracking
        temporal_setting (str): Temporal setting for DEVA
        size (int): Size parameter for DEVA
        mem_every (int): Memory savings frequency for DEVA
        detection_every (int): Detection frequency for DEVA
        max_num_objects (int): Maximum number of objects to track
    """
    # Create output directory structure: session_name/camera_view_name/
    output_dir = base_output_dir / session_name / camera_view_name
    
    # Skip if already processed
    if (output_dir / 'pred.json').exists():
        print(f"Skipping already processed {session_name}/{camera_view_name}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {session_name}/{camera_view_name}")
    
    # Verify directories exist and have content
    print(f"  Camera directory: {camera_dir} (exists: {camera_dir.exists()})")
    if camera_dir.exists():
        frames = list(camera_dir.glob('*.jpg')) + list(camera_dir.glob('*.png'))
        print(f"  Found {len(frames)} frames in camera directory")
    
    print(f"  Output directory: {output_dir}")
    
    # Run DEVA tracking
    print(f"  Starting DEVA tracking for {session_name}/{camera_view_name}...")
    success = tracker.run_tracking(
        img_path=str(camera_dir),
        output_path=str(output_dir),
        chunk_size=chunk_size,
        temporal_setting=temporal_setting,
        size=size,
        mem_every=mem_every,
        detection_every=detection_every,
        max_num_objects=max_num_objects
    )
    
    if success:
        print(f"  Finished DEVA tracking for {session_name}/{camera_view_name}")
    else:
        print(f"  Failed DEVA tracking for {session_name}/{camera_view_name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process video sequences for DEVA tracking with support for sessions and camera views")
    
    # Base directories
    parser.add_argument("--frames_dir", type=str, 
                       default="/nas/project_data/B1_Behavior/rush/kaan/first_play_local/models/memflow/demo_input_images",
                       help="Directory containing input frames organized by session/camera_view")
    parser.add_argument("--output_dir", type=str, 
                       default="/nas/project_data/B1_Behavior/rush/kaan/first_play_local/data/tests/deva",
                       help="Base output directory for results")
    
    # Processing options
    parser.add_argument("--gpu_id", type=int, default=0,
                      help="GPU ID to use for processing")
    parser.add_argument("--num_parts", type=int, default=1,
                      help="Number of parts to split the session dataset into")
    parser.add_argument("--process_part", type=int, default=0,
                      help="Which part to process (0-indexed, must be less than num_parts)")
    
    # Session and camera view options
    parser.add_argument("--session", type=str, default=None,
                      help="Specific session to process (None for all)")
    parser.add_argument("--camera_view", type=str, default=None,
                      help="Specific camera view to process (None for all)")
    
    # DEVA tracking options
    parser.add_argument("--chunk_size", type=int, default=8,
                      help="Chunk size for DEVA tracking")
    parser.add_argument("--temporal_setting", type=str, default="semionline",
                      help="Temporal setting for DEVA ('semionline' or 'online')")
    parser.add_argument("--size", type=int, default=-1,
                      help="Size parameter for DEVA (-1 for original size)")
    parser.add_argument("--mem_every", type=int, default=5,
                      help="Memory savings frequency for DEVA")
    parser.add_argument("--detection_every", type=int, default=10,
                      help="Detection frequency for DEVA")
    parser.add_argument("--max_num_objects", type=int, default=100,
                      help="Maximum number of objects to track")
    
    args = parser.parse_args()
    
    # Run the video processing
    process_videos(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        num_parts=args.num_parts,
        process_part=args.process_part,
        session=args.session,
        camera_view=args.camera_view,
        chunk_size=args.chunk_size,
        temporal_setting=args.temporal_setting,
        size=args.size,
        mem_every=args.mem_every,
        detection_every=args.detection_every,
        max_num_objects=args.max_num_objects
    )
