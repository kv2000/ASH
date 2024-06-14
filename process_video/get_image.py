"""
@File: get_image.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2024-6-12
@Desc: It extracts the images, and foreground mask it. 
The resolution is set to 1285 x 940 by default, similar to the DynaCap Dataset
GLHF & :D
"""

import os
import sys
import cv2 as cv
from PIL import Image

from av import time_base as AV_TIME_BASE
import av
import av.datasets
import numpy as np
import imghdr
import argparse

def get_keyframe_interval(cap):
    
    frame_number = 0   
    
    fps = cap.streams.video[0].average_rate

    video_stream = cap.streams.video[0]
   
    assert int(1 / video_stream.time_base) % fps == 0  
   
    offset_timestamp = int(1 / video_stream.time_base / fps)
   
    video_stream.codec_context.skip_frame = "NONKEY"

    target_timestamp = int((frame_number * AV_TIME_BASE ) / video_stream.average_rate)

    cap.seek(target_timestamp)
    
    result  = []
    iter = 0
    
    for frame in cap.decode(video_stream):
        #print(frame)	
        if(iter>1):
            video_stream.codec_context.skip_frame = "DEFAULT"		
            return result[1] - result[0]
            break	   
        result.append(int(frame.pts /  offset_timestamp))
        iter+=1 
    
    video_stream.codec_context.skip_frame = "DEFAULT"	
    
    return -1

def gao(input_img_dir = None, input_mask_dir = None, 
        camera_id = 0, 
        output_img_dir = None, output_mask_dir = None,
        sample_start_id = None, sample_interval = None,
        sample_end_id = None, no_scale_image = None
    ):

    num_threshold = 200
    num_threshold_big = 150000

    padded_id = str(camera_id).zfill(3)
    print('padded id', padded_id)
    input_image_name = os.path.join(input_img_dir, 'stream' + padded_id + '.mp4')
    input_mask_name = os.path.join(input_mask_dir, 'stream' + padded_id + '.mp4')

    print(input_image_name, input_mask_name)

    if not os.path.isfile(input_image_name):
        return
    if not os.path.isfile(input_mask_name):
        return
    
    cap_img = av.container
    cap_img = av.open(av.datasets.curated(input_image_name))
    cap_img.streams.video[0].thread_count = 16

    cap_mask = av.container
    cap_mask = av.open(av.datasets.curated(input_mask_name))
    cap_mask.streams.video[0].thread_count = 16

    os.makedirs(
        os.path.join(output_img_dir, str(camera_id)), exist_ok=True        
    )
    os.makedirs(
        os.path.join(output_mask_dir, str(camera_id)), exist_ok=True        
    )

    total_frames = cap_img.streams.video[0].frames
    print('number of frames', total_frames, flush=True)

    for i in range(total_frames):
        print('processing', i, flush=True)
        
        img = None 
        for img in cap_img.decode(video=0):
            break
        
        mask = None 
        for mask in cap_mask.decode(video=0):
            break
    
        if not ((i % sample_interval) == sample_start_id):
            continue 

        if not (i%10 == 0):
            continue
    
        if i > sample_end_id:
            break

        output_image_name = os.path.join(
            output_img_dir , str(camera_id ), 'image_c_' + str(camera_id ) + '_f_' + str(i) + '.png' 
        )

        output_mask_name = os.path.join(
            output_mask_dir , str(camera_id ), 'image_c_' + str(camera_id ) + '_f_' + str(i) + '.png' 
        )

        if (os.path.isfile(output_image_name) and os.path.isfile(output_mask_name)):
            if ((not (imghdr.what(output_image_name) is None)) and (not (imghdr.what(output_mask_name) is None))):
                continue
            else:
                print('broken image', output_image_name)

        cur_img = np.array(img.to_image())
        cur_mask = np.array(mask.to_image())
               
        cur_mask = np.asarray(cur_mask)
        
        cur_mask = (cur_mask > 50).astype('uint8')[...,-1]

        countors, hierarchy = cv.findContours(
            cur_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
        )

        nb_components = len(countors)

        if nb_components == 0:
            fin_mask = np.zeros([cur_img.shape[0], cur_img.shape[1]])
        elif nb_components == 1:
            fin_mask = cur_mask
        else:
            area = []
            for j in range(len(countors)):
                area.append(cv.contourArea(countors[j]))
            
            max_idx = np.argmax(area)

            fin_mask = cur_img.copy()

            for k in range(len(countors)):
                if k != max_idx:
                    cv.fillPoly(fin_mask, [countors[k]], 0)

        kernel = np.ones((3, 3), np.uint8)
        kernel1 = np.ones((6, 6), np.uint8)
        
        cur_mask = cv.erode(cur_mask,kernel1)
        cur_mask = cv.dilate(cur_mask,kernel)
        
        if not no_scale_image:
            cur_mask = cv.resize(cur_mask, dsize=[1285, 940], interpolation=cv.INTER_AREA)
            cur_img = cv.resize(cur_img, dsize=[1285, 940], interpolation=cv.INTER_AREA)

        cur_img = cur_img * cur_mask[...,None]

        Image.fromarray((cur_img).astype(np.uint8)).save(output_image_name)
        Image.fromarray((cur_mask * 255).astype(np.uint8)).save(output_mask_name)

    return []

if __name__ == '__main__':
    print('wootwootwo')

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id',         type=int,                help='The video camera id')
    parser.add_argument('--sample_start_id',   type=int, default= 0,    help='The starting frame id')
    parser.add_argument('--sample_end_id',     type=int, default= 25000,help='The end frame id')
    parser.add_argument('--sample_interval',   type=int, default= 10,   help='Sampel for every X frames')
    
    parser.add_argument('--input_video_dir',   type=str)
    parser.add_argument('--input_mask_dir',    type=str)
    parser.add_argument('--output_img_dir',    type=str)
    parser.add_argument('--output_mask_dir',   type=str)
    
    parser.add_argument('--no_scale_img',      type=int,  default=0,    help='Scaled version is 1285x940, no scaled is the original video')
    

    args = parser.parse_args()

    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_mask_dir, exist_ok=True)

    gao(
        input_img_dir = args.input_video_dir,
        input_mask_dir = args.input_mask_dir,
        camera_id = args.camera_id,
        output_img_dir = args.output_img_dir,
        output_mask_dir = args.output_mask_dir,
        sample_interval = args.sample_interval,
        sample_start_id = args.sample_start_id,
        sample_end_id = args.sample_end_id,
        no_scale_image = args.no_scale_img
    )   
    
    