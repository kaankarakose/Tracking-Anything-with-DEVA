import sys
import os
from os import path
from argparse import ArgumentParser


#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
#torch.cuda.set_device(0)

import gc
gc.collect()


from torch.utils.data import DataLoader
import numpy as np
sys.path.append('/home/kaan/Projects/object_manupilation/track_anything_deva/Tracking-Anything-with-DEVA')
import time
#print(sys.path.remove('/home/kaan/Projects/object_manupilation/deva/Tracking-Anything-with-DEVA'))

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args
from deva.ext.automatic_sam import get_sam_model
from deva.ext.automatic_processor import process_frame_automatic as process_frame

from tqdm import tqdm
import json
def print_cuda_memory_usage(line):
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(f"\nLine : {line} Memory Allocated: {allocated} bytes, Memory Cached: {cached} bytes")

if __name__ == '__main__':
    #torch.cuda.memory._record_memory_history(True) # 
    # enable memory history, which will
    # add tracebacks and event history to snapshots
    torch.autograd.set_grad_enabled(False)
    ## RUN COMMAND python /home/kaan/Projects/object_manupilation/track_anything_deva/Tracking-Anything-with-DEVA/demo/demo_automatic.py --chunk_size 4 --img_path /nas/project_data/B1_Behavior/rush/object-manipulation/segmentation/frames --amp --temporal_setting semionline  --size -1 --output /nas/project_data/B1_Behavior/rush/object-manipulation/segmentation/deva_output
    # for id2rgb
#python /home/kaan/Projects/object_manupilation/track_anything_deva/Tracking-Anything-with-DEVA/demo/demo_automatic.py --chunk_size 2 --img_path /nas/project_data/B1_Behavior/rush/object-manipulation/make_believe/segmentation/dataset_frames/Argo5 --amp --temporal_setting semionline  --size 360 --output /nas/project_data/B1_Behavior/rush/object-manipulation/make_believe/segmentation/deva_output --save_all --max_num_objects 5
    np.random.seed(42)
    """
    Arguments loading
    """
    parser = ArgumentParser()
    add_common_eval_args(parser)

    add_ext_eval_args(parser)

    add_auto_default_args(parser)
  
    deva_model, cfg, args = get_model_and_config(parser)
    # print(f"Variant_{args.sam_variant}_chunk_{args.chunk_size}_MNO{args.max_num_objects}_size_{args.size}.pickle")
    # torch.cuda.memory._dump_snapshot(f"Variant_{args.sam_variant}_chunk_{args.chunk_size}_MNO{args.max_num_objects}_size_{args.size}.pickle")
    # raise ValueError
    #print_cuda_memory_usage("First_Allocate")
    sam_model = get_sam_model(cfg, 'cuda')
    

    """
    Temporal setting
    """
    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    # get data
    video_reader = SimpleVideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=2)
    out_path = cfg['output']

    # Start eval
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    cfg['enable_long_term_count_usage'] = (
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements'])
   
    #print('Configuration:', cfg)

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = args.num_voting_frames - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)
    #print_cuda_memory_usage(86)
    
    torch.cuda.empty_cache()
    tick_all = time.time()
    with torch.cuda.amp.autocast(enabled=args.amp):
        for ti, (frame, im_path) in enumerate(loader):
            #tickf = time.time()
            process_frame(deva, sam_model, im_path, result_saver, ti, image_np=frame)
            #tackf = time.time()
            #print(f"Frame : {ti} Time took",tackf - tickf
            #torch.cuda.empty_cache()
        flush_buffer(deva, result_saver)
    # tack_all = time.time()
    # print(tack_all - tick_all)
    # result_saver.end()
    
    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json

    


   # torch.cuda.empty_cache()
   # torch.cuda.memory_snapshot(f"Variant_{args.sam_variant}_chunk_{args.chunk_size}_MNO{args.max_num_objects}_size_{args.size}.pickle")






