import os
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("./DeepCharacters_Pytorch/")

import time
from icecream import ic
import argparse
from pyhocon import ConfigFactory
import numpy as np
import cv2 as cv
import trimesh
from tqdm import tqdm
import math
from icecream import ic
import pickle as pkl
from PIL import Image
from einops import rearrange

from typing import NamedTuple
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn

from models.dataset_no_charactor import ASH_Inference_Dataset
import torchvision


from argparse import ArgumentParser, Namespace

#############################################################################################

from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from gaussian_renderer import render
from scene import GaussianModel

from scene.unet_2 import UNet as unet_sh
from scene.unet import UNet as unet_geo


###################################################################################################################
#                                        Some Adapted Modules from the Gaussian                                   #
###################################################################################################################

def readCamerasFromTransforms_from_pang(transformsfile):
    """
        From Haokai for Adapting to DynaCap dataset.
    """
    cam_infos = []

    with open(os.path.join(transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]

        for idx, frame in enumerate(frames):
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = np.transpose(matrix[:3, :3])
            R[:, 0] = R[:, 0]
            T = matrix[:3, 3]

            FovY = frame["camera_angle_y"]
            FovX = frame["camera_angle_x"]

            Cy = frame["cy"]
            Cx = frame["cx"]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, Cy=Cy, Cx=Cx, image=None,
                                        image_path=None, image_name=None, width=None,
                                        height=None))
            
    return cam_infos

class CameraInfo(NamedTuple):
    """
        From Haokai for Adapting to DynaCap dataset.
    """
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    Cy: np.array
    Cx: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class Camera(nn.Module):
    def __init__(self, uid, R, T, FoVx, FoVy, Cx, Cy,
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device="cpu",
                 image_width = None,
                 image_height = None,
                 ):
        super(Camera, self).__init__()

        """
            From Haokai for Adapting to DynaCap dataset.
        """
        
        self.uid = uid

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.Cx = Cx
        self.Cy = Cy

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
            Cx=self.Cx, Cy=self.Cy,
            width=self.image_width, height=self.image_height
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class Mock_PipelineParams:
    def __init__(self):
        """
           Nothing but a hack 
        """
        self.convert_SHs_python = False
        self.compute_cov3D_python = False

#############################################################################################

class Runner:
    def __init__(self, conf):
        ###################################################################################################################
        #                                        General Training/Testing Initialization                                  #
        ###################################################################################################################
        # Also plzzz check out the configurations files
        self.conf = conf
        self.device = self.conf['general']['device']
        self.device0 = self.conf['general']['device_dataset']
        
        self.gaussian_camera_json_dir = self.conf['dataset']['camera_json_dir']
        self.gaussian_checkpoint_dir = self.conf['train']['gaussian_checkpoint_dir']
        
        self.img_width = self.conf.get_int('dataset.img_width')
        self.img_height = self.conf.get_int('dataset.img_height')
        
        self.is_white_background = self.conf.get_bool('dataset.is_white_background')
        
        if self.is_white_background:
            self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device)
        else:
            self.background = torch.tensor([0., 0., 0.], dtype=torch.float32).to(self.device)
            
        self.dst_dir = self.conf['general']['output_dir']     
        os.makedirs(self.dst_dir, exist_ok=True)   

        ###################################################################################################################
        # good effort from haokai to adapt for the dynacap dataset
        self.camera_arr = readCamerasFromTransforms_from_pang(self.gaussian_camera_json_dir)
        self.real_camera_arr = None
        self.get_real_cameras()
  
        self.mocked_pipeline = Mock_PipelineParams()
                 
        self.dataset = ASH_Inference_Dataset(
            self.conf, device = self.device0
        )
             
        self.non_empty_idx = torch.LongTensor(self.dataset.uv_idx_np).to(self.device)
        self.non_empty_idy = torch.LongTensor(self.dataset.uv_idy_np).to(self.device)

        ###################################################################################################################
   
        self.unet_appearance = unet_sh(18, 48, 6).to(self.device)
        self.unet_geometry = unet_geo(18, 11).to(self.device)
        self.gaussians = GaussianModel(sh_degree = self.conf.get_int('model.gaussian.sh_degree'))
        
        ###################################################################################################################
        
        self.load_checkpoints()
        
    def load_checkpoints(self):
        
        print('++++++ start loading checkpoints')
        
        if os.path.isfile(self.gaussian_checkpoint_dir):
            cur_state_dict = torch.load(self.gaussian_checkpoint_dir, map_location=self.device)
            
            self.unet_appearance.load_state_dict(cur_state_dict['unet_appearance'])
            self.unet_geometry.load_state_dict(cur_state_dict['unet_geometry'])
            
            if (self.unet_appearance is not None) and ('unet_appearance' in cur_state_dict.keys()):
                print('+++++ loading checkpoints unet_appearance')
                self.unet_appearance.load_state_dict(cur_state_dict['unet_appearance'])     

            if (self.unet_geometry is not None) and ('unet_geometry' in cur_state_dict.keys()):
                print('+++++ loading checkpoints unet_geometry')
                self.unet_geometry.load_state_dict(cur_state_dict['unet_geometry'])     
            
        else:
            print('++++++ checkpoint not found:', self.gaussian_checkpoint_dir)
                
        print('++++++ end loading checkpoints')
        
        return 

    def get_real_cameras(self):

        self.real_camera_arr = []

        for i in range(len(self.camera_arr)):
            cur_camera_info = Camera(
                uid = i,
                R = self.camera_arr[i].R.copy(),
                T = self.camera_arr[i].T.copy(),
                FoVx = self.camera_arr[i].FovX, 
                FoVy = self.camera_arr[i].FovY,
                Cx = self.camera_arr[i].Cx,
                Cy = self.camera_arr[i].Cy,
                image_height = self.img_height,
                image_width = self.img_width
            )
            self.real_camera_arr.append(cur_camera_info)
            
        return 
      
    def render_frame(self, frame_id, camera_id = 18):
        
        with torch.no_grad(): 
            
            ret_dict = self.dataset.get_val_img_dict(frame_id)                 
            
            cur_frame_id = ret_dict['cur_frame_id']
            transform_vec = ret_dict['transform_vec']
            ddc_cond_map = ret_dict['ddc_cond_map']
            skeletal_joints = ret_dict['skeletal_joints']
            
            canonincal_pos_map = ret_dict['canoincal_pos_map']
            posed_normal_map = ret_dict['posed_normal_map']
            frame_global_translation = ret_dict['frame_global_translation']
            
            transform_vec = torch.FloatTensor(transform_vec).to(self.device)       
            ddc_cond_map = torch.FloatTensor(ddc_cond_map).permute(2, 0, 1).unsqueeze(0).to(self.device) / 1000.0
            canonincal_pos_map = torch.FloatTensor(canonincal_pos_map[0]).to(self.device) / 1000.0
            posed_normal_map = torch.FloatTensor(posed_normal_map).permute(2, 0, 1).unsqueeze(0).to(self.device)
            frame_global_translation = torch.FloatTensor(frame_global_translation).unsqueeze(0).to(self.device)
            network_motion_cond = torch.cat([ddc_cond_map, posed_normal_map], dim=1)
                    
            ###############################################################################################################
            
            geo_feats = self.unet_geometry(network_motion_cond).squeeze(0)        
            app_feats = self.unet_appearance(network_motion_cond, frame_global_translation).squeeze(0)
             
            geo_feats = geo_feats[:,self.non_empty_idx, self.non_empty_idy].permute(1, 0).contiguous()
            app_feats = app_feats[:,self.non_empty_idx, self.non_empty_idy].permute(1, 0).contiguous()
            
            canonical_delta = geo_feats[:,8:]
            canonincal_pos_map =  canonincal_pos_map + canonical_delta  
            transform_vec[:, :3, 3] = transform_vec[:, :3, 3] / 1000.

            ###############################################################################################################

            pos_xyz = self.compute_deformed_template(canonincal_pos_map, transform_vec)
            
            self.gaussians._xyz = pos_xyz[:, [2, 0, 1]]
            self.gaussians._scaling = geo_feats[:, :3]
            self.gaussians._rotation = geo_feats[:, 3:7]
            self.gaussians._opacity = geo_feats[:, 7].unsqueeze(1)
            
            self.gaussians._features_dc = app_feats[:, :3].reshape(-1, 1, 3)
            self.gaussians._features_rest = app_feats[:, 3:48].reshape(-1, 15, 3)

            camera_info = Camera(
                uid = cur_frame_id,
                R = self.camera_arr[camera_id].R.copy(),
                T = self.camera_arr[camera_id].T.copy(),
                FoVx = self.camera_arr[camera_id].FovX, 
                FoVy = self.camera_arr[camera_id].FovY,
                Cx = self.camera_arr[camera_id].Cx,
                Cy = self.camera_arr[camera_id].Cy,
                image_height = self.img_height,
                image_width = self.img_width
            )

            camera_info.world_view_transform = camera_info.world_view_transform.to(self.device)
            camera_info.projection_matrix = camera_info.projection_matrix.to(self.device)
            camera_info.full_proj_transform = camera_info.full_proj_transform.to(self.device)
            camera_info.camera_center = camera_info.camera_center.to(self.device)
    
            ###############################################################################################################
            
            render_pkg = render(camera_info, self.gaussians, self.mocked_pipeline, self.background)
                                
            image = render_pkg["render"]
            
            #"""
            img_fine = image.detach().cpu()
            img_fine = rearrange(img_fine,'c x y -> x y c')
            img_fine = np.clip(img_fine.numpy(), 0, 1) * 255

            ret_img = Image.fromarray(img_fine.astype(np.uint8))             
            ret_img.save(os.path.join(
                self.dst_dir, str(camera_id), str(frame_id) + '.png'
            ))
            #"""
                    
        return {}

    def compute_deformed_template(self, can_tex, can_trans):
        v = can_tex
        v = torch.cat([v, torch.ones(v.shape[0], 1).to(self.device)], dim=-1)
        t = can_trans

        tv = torch.matmul(t, v.unsqueeze(-1)).squeeze(-1)
        tv = tv[:, :3] / tv[:, 3:]
        return tv
    
    def dump_imgs(self):
        
        print('++++++ start dumping images')
        
        print(self.dataset.val_camera)
        print(self.dataset.val_frame_idx)
        
        for each_cam_id in self.dataset.val_camera:
            os.makedirs(os.path.join(
                self.dst_dir, str(each_cam_id)
            ), exist_ok=True)
                
        for i in tqdm(range(len(self.dataset.val_camera) * len(self.dataset.val_frame_idx))):
            self.render_frame(
                frame_id= self.dataset.val_frame_idx[i // len(self.dataset.val_camera)], 
                camera_id= self.dataset.val_camera[i % len(self.dataset.val_camera)]
            )
            
        return 
        
if __name__ == '__main__':
    print('wootwootwo')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/Subject0022/subject0022_val.conf')
    args = parser.parse_args()

    f = open(args.conf)
    conf_text = f.read()
    f.close()
    
    preload_conf = ConfigFactory.parse_string(conf_text)

    runner = Runner(
        preload_conf
    )
    
    runner.dump_imgs()