"""
@File: exp_trainer_base.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2024-6-12
@Desc: Simplified version of the training. Cuz u don't need to fit individual frames for initialization;
but by fitting the gaussians to the skinned mesh 
"""

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

from models.dataset_no_charactor_train import ASH_Train_Dataset
import torchvision


from argparse import ArgumentParser, Namespace

#############################################################################################
from torch.utils.tensorboard import SummaryWriter
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, BasicPointCloud

from gaussian_renderer import render
from scene import GaussianModel

from scene.unet_2 import UNet as unet_sh
from scene.unet import UNet as unet_geo

import lpips
import skimage

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
    def __init__(self, conf, mode='train', is_continue=True):
        ###################################################################################################################
        #                                        General Training/Testing Initialization                                  #
        ###################################################################################################################
        # Also plzzz check out the configurations files
        self.conf = conf
        self.device = self.conf['general']['device']
        self.device0 = self.conf['general']['device_dataset']
        self.is_continue = is_continue
        
        self.gaussian_camera_json_dir = self.conf['dataset']['camera_json_dir']
        
        self.is_white_background = self.conf.get_bool('dataset.is_white_background')
        
        if self.is_white_background:
            self.background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device)
        else:
            self.background = torch.tensor([0., 0., 0.], dtype=torch.float32).to(self.device)
            
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
    
        self.dataset = ASH_Train_Dataset(
            self.conf, device = self.device0
        ) 
        
        ###################################################################################################################
        # good effort from haokai to adapt for the dynacap dataset
        self.camera_arr = readCamerasFromTransforms_from_pang(self.gaussian_camera_json_dir)
  
        self.mocked_pipeline = Mock_PipelineParams()

        self.non_empty_idx = torch.LongTensor(self.dataset.uv_idx_np).to(self.device)
        self.non_empty_idy = torch.LongTensor(self.dataset.uv_idy_np).to(self.device)

        self.org_train_list = [i for i in self.dataset.chosen_frame_id_list]
        self.mock_train_list = [i for i in range(110, 1000, 10)]

        ###################################################################################################################
        #                                        model related                                                            #
        ################################################################################################################### 
   
        self.unet_appearance = unet_sh(18, 48, 6).to(self.device)
        self.unet_geometry = unet_geo(18, 11).to(self.device)
        self.gaussians = GaussianModel(sh_degree = self.conf.get_int('model.gaussian.sh_degree'))
        self.gaussian = GaussianModel(sh_degree = self.conf.get_int('model.gaussian.sh_degree'))
        
        ###################################################################################################################
        #                                        tranining related                                                        #
        ################################################################################################################### 
        
        self.first_phase_end = 15000
        
        self.iter_step = 0
        self.from_scratch = True
        self.end_iter = self.conf.get_int('train.end_iter', default=30000)
        self.worker_num = self.conf.get_int('dataset.worker_num')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.plot_histogram_freq = self.conf.get_int('train.plot_histogram_freq')

        self.warm_up_end = self.conf.get_int('train.warm_up_end', default=1000)
        self.start_constant_lr = self.conf.get_int('train.start_constant_lr', default=30000)
        self.learning_rate = self.conf.get_float('train.learning_rate', default=5e-4)
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha', default=0.04)

        self.color_weight = self.conf.get_float('train.color_weight', default=0.8)
        self.ssim_weight = self.conf.get_float('train.ssim_weight', default=0.2)

        self.params_to_train = []
        self.get_trainable_variables()
        self.optimizer = torch.optim.Adam(self.params_to_train, lr=self.learning_rate, eps=1e-15)

        self.latest_model_name = None
        self.updated_loss_dict = self.make_tensorboard_settings()
 
        self.summary_writer = SummaryWriter(
            self.updated_loss_dict['tensorboard_dir'] + '/logs'
        )
        
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device) # best forward scores
        self.loss_fn_alex.eval()
        
        if self.is_continue:
            self.get_latest_checkpoints()
        else:
            print('+++++ start from scratch')
        
        self.train()

    def get_latest_checkpoints(self):
        print('+++++ run load checkpoints')
        # fin the newest
        if os.path.isdir(os.path.join(self.base_exp_dir, 'checkpoints')):
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_id_list = []
            # find the last checkpoint, id is non-filled
            for each_model_name in model_list_raw:
                model_id_list.append(
                    int(each_model_name)
                )
            model_id_list.sort()
            print('+++++ avilable model id ', model_id_list)
            if len(model_id_list) > 0:
                self.latest_model_name = str(model_id_list[-1])
                self.load_checkpoint()
            else:
                print('+++++ no checkpoints, start from scratch')
        else:
            print('+++++ not even checkpint folder, start from scratch')

        print('+++++ end load checkpoints')
        return
        
    def load_checkpoint(self):
        print('+++++ loading checkpoints from:', self.latest_model_name)
        
        fin_checkpoint_file_name = os.path.join(
            self.base_exp_dir, 'checkpoints', self.latest_model_name, 'state_dict.pth'
        )

        cur_state_dict = torch.load(fin_checkpoint_file_name, map_location=self.device)
        
        self.iter_step = cur_state_dict['iter_step'] + 1
                
        if (self.optimizer is not None) and ('optimizer' in cur_state_dict.keys()) :
            print('+++++ loading checkpoints optimizer')
            self.optimizer.load_state_dict(cur_state_dict['optimizer'])
        
        if (self.unet_appearance is not None) and ('unet_appearance' in cur_state_dict.keys()) :
            print('+++++ loading checkpoints unet_appearance')
            self.unet_appearance.load_state_dict(cur_state_dict['unet_appearance'])

        if (self.unet_geometry is not None) and ('unet_geometry' in cur_state_dict.keys()) :
            print('+++++ loading checkpoints unet_geometry')
            self.unet_geometry.load_state_dict(cur_state_dict['unet_geometry'])

        print('+++++ end loading checkpoints') 

    def get_trainable_variables(self):
        print('+++++ start to get trainalbe variables')
        self.params_to_train = []

        self.params_to_train += [{'params':self.unet_appearance.parameters(),'name':'unet_appearance'}] \
            + [{'params':self.unet_geometry.parameters(),'name':'unet_geometry'}]       

        print('+++++ end to get trainalbe variables')
        return

    def update_learning_rate(self):

        if self.iter_step < self.warm_up_end:
            learning_factor = (self.iter_step + 1) / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (
                min(self.iter_step, self.start_constant_lr + 2000) - self.warm_up_end
            ) / (
                min(self.end_iter, self.start_constant_lr + 2000) - self.warm_up_end
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        
        weighted_learing_rate = self.learning_rate * learning_factor
        
        if self.iter_step > self.start_constant_lr:
            weighted_learing_rate =  alpha * self.learning_rate

        for g in self.optimizer.param_groups:
            g['lr'] = weighted_learing_rate
        
        return 
    
    def profile_loss(self, loss_dict):
        
        fin_loss_dict = {}
        
        for each_key in self.updated_loss_dict['loss_dict']:
            if each_key in loss_dict:
                fin_loss_dict[each_key] = loss_dict[each_key]
            else:
                fin_loss_dict[each_key] = 0.
        
        for each_key in fin_loss_dict:
            self.summary_writer.add_scalar(each_key, fin_loss_dict[each_key], self.iter_step)
    
        return 

    def make_tensorboard_settings(self):
        ret_dict = {}
        ret_dict['tensorboard_dir'] = os.path.join(
            self.base_exp_dir, 'exp_stats'
        )
        
        loss_dict = {
            'loss': True,
            'color': True, 
            'ssim':True,
            'lr': True,
            'dc': True,
            'rest': True,
            'rot': True,
            'op': True,
            'del': True
        }

        ret_dict['loss_dict'] = loss_dict
        
        print("#################tensorboard settings###################")
        ic(ret_dict)
        print("########################################################")
        return ret_dict
 
    def save_checkpoint(self):
        print('+++++ saving checkpoints:', self.iter_step)
        cur_base_dir = os.path.join(self.base_exp_dir, 'checkpoints', str(self.iter_step))
        state_dict_file_name = os.path.join(
            cur_base_dir, 'state_dict.pth'
        )
        
        os.makedirs(cur_base_dir, exist_ok=True)
        
        cur_state_dict = {
            'iter_step': self.iter_step
        }
        
        if self.optimizer is not None:
            print('+++++ saving checkpoints optimizer')
            cur_state_dict['optimizer'] = self.optimizer.state_dict()
            
        if self.unet_appearance is not None:
            print('+++++ saving checkpoints unet_appearance')
            cur_state_dict['unet_appearance'] = self.unet_appearance.state_dict()

        if self.unet_geometry is not None:
            print('+++++ saving checkpoints unet_geometry')
            cur_state_dict['unet_geometry'] = self.unet_geometry.state_dict()

        torch.save(
            cur_state_dict, state_dict_file_name
        )

        print('+++++ end saving checkpoints:')  

    def plot_weight_histogram(self):

        print('+++++ Start plotting the histogram for iteration', self.iter_step)

        for name, param in self.unet_appearance.named_parameters():
            cur_name = 'unet_appearance/' + name
            self.summary_writer.add_histogram(
                cur_name, param.cpu(), self.iter_step
            )

        for name, param in self.unet_geometry.named_parameters():
            cur_name = 'unet_geometry/' + name
            self.summary_writer.add_histogram(
                cur_name, param.cpu(), self.iter_step
            )

        return 

    def compute_deformed_template(self, can_tex, can_trans):
        v = can_tex
        v = torch.cat([v, torch.ones(v.shape[0], 1).to(self.device)], dim=-1)
        t = can_trans

        tv = torch.matmul(t, v.unsqueeze(-1)).squeeze(-1)
        tv = tv[:, :3] / tv[:, 3:]
        return tv

    def compute_deformed_template_np(self, can_tex, can_trans):
        v = can_tex
        v = np.concatenate([v, np.ones([v.shape[0], 1])], axis=-1)
        t = can_trans

        tv = np.matmul(t, v[...,None])[...,0]
        tv = tv[:, :3] / tv[:, 3:]

        return tv

    def create_ref_gaussian(self, ret_dict):
        
        pt_normal = ret_dict['posed_normal_map'][self.dataset.uv_idx_np, self.dataset.uv_idy_np,:3]
        # mock color from the skinned mesh
        pt_color = self.dataset.real_texture_map[(self.dataset.barycentric_tex_size - 1 - self.dataset.uv_idy_np), self.dataset.uv_idx_np,:3]
        pt_trans = ret_dict['transform_vec']
        pt_trans[:, :3, 3] = pt_trans[:, :3, 3] / 1000.
        
        pt_pos = self.compute_deformed_template_np(
            ret_dict['canoincal_pos_map'][0] / 1000., pt_trans
        )
                
        ret_ref_gaussian = GaussianModel(
            sh_degree = self.conf.get_int('model.gaussian.sh_degree')
        )
        
        ret_ref_pcd = BasicPointCloud(
            points = pt_pos[:, [2, 0, 1]].copy(),
            normals = pt_normal[:, [2, 0, 1]].copy(),
            colors = pt_color.copy()
        )
            
        ret_ref_gaussian.create_from_pcd(
            ret_ref_pcd, 0.0
        )
        
        ret_ref_gaussian._features_rest = ret_ref_gaussian._features_rest * 0.
                
        return ret_ref_gaussian

    def mae(self, imageA, imageB):
        err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
        err /= float(imageA.shape[0] * imageA.shape[1]* imageA.shape[2])
        return err

    def mse(self,imageA, imageB):

        errImage = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2,2)
        errImage = np.sqrt(errImage)

        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1]* imageA.shape[2])

        return err,errImage

    # compute the curves
    def val_img_loss(self, output_img, gt_img, gt_mask):
                
        H = 600
        W = 496
        
        g = gt_img.astype(np.float32) / 255.0
        t = output_img.astype(np.float32) / 255.0
        
        h, w = g.shape[0], g.shape[1]

        kernel = np.ones((3, 3), np.uint8)
        imgMask = cv.erode(gt_mask, kernel)
        imgMask = cv.resize(imgMask, (t.shape[1], t.shape[0]))
        
        imgMask = imgMask[...,None]

        g = g * imgMask + (1.0 - imgMask)
        t = t * imgMask + (1.0 - imgMask)

        ii, jj = np.where(~(t == 1).all(-1))

        hmin, hmax = np.min(ii), np.max(ii)
        uu = (H - (hmax + 1 - hmin)) // 2
        vv = H - (hmax - hmin) - uu
        
        if hmin - uu < 0:
            hmin, hmax = 0, H
        elif hmax + vv > h:
            hmin, hmax = h - H, h
        else:
            hmin, hmax = hmin - uu, hmax + vv

        # bounds for U direction
        wmin, wmax = np.min(jj), np.max(jj)
        uu = (W - (wmax + 1 - wmin)) // 2
        vv = W - (wmax - wmin) - uu
        if wmin - uu < 0:
            wmin, wmax = 0, W
        elif wmax + vv > w:
            wmin, wmax = w - W, w
        else:
            wmin, wmax = wmin - uu, wmax + vv

        g_cu = torch.FloatTensor(g).to(self.device) * 2 - 1.0
        t_cu = torch.FloatTensor(t).to(self.device) * 2 - 1.0

        g = g[hmin: hmax, wmin: wmax]
        t = t[hmin: hmax, wmin: wmax]

        g_cu = rearrange(g_cu, 'h w c -> c h w')
        t_cu = rearrange(t_cu, 'h w c -> c h w')

        g_cu = g_cu.unsqueeze(0)
        t_cu = t_cu.unsqueeze(0)

        d = torch.mean((self.loss_fn_alex(g_cu, t_cu))).detach().cpu().numpy()

        mseValue, errImg= self.mse(g, t)
        maeValue = self.mae(g, t)
        
        psnr = 10 * np.log10((1 ** 2) / mseValue)
        ssims = skimage.metrics.structural_similarity(g, t, channel_axis=2, data_range=1)

        return psnr, ssims, maeValue, mseValue, d

    def render_frame(self, frame_id, camera_id = 18):
        
        with torch.no_grad(): 
            
            ret_dict = self.dataset.get_val_img_dict(frame_id, camera_id)                 
            
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
                image_height = self.dataset.img_height,
                image_width = self.dataset.img_width
            )

            camera_info.world_view_transform = camera_info.world_view_transform.to(self.device)
            camera_info.projection_matrix = camera_info.projection_matrix.to(self.device)
            camera_info.full_proj_transform = camera_info.full_proj_transform.to(self.device)
            camera_info.camera_center = camera_info.camera_center.to(self.device)
    
            ###############################################################################################################
            
            render_pkg = render(camera_info, self.gaussians, self.mocked_pipeline, self.background)
                                
            image = render_pkg["render"]
            
            img_fine = image.detach().cpu()
            img_fine = rearrange(img_fine,'c x y -> x y c')  
                    
        return {
            'output': img_fine.numpy(),
            'gt_img': ret_dict['ret_img'],
            'gt_mask': ret_dict['ret_mask']
        }

    def validate_image(self):
        
        print('+++++ start validate image ', self.dataset.val_frame_idx, self.dataset.val_camera)
        
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        
        psnrs = []
        ssims = []
        maes = []
        mses = []
        lpips_val = []

        # render each frame and ret the numbers
        for frame_id in self.dataset.val_frame_idx:
            for camera_id in self.dataset.val_camera:
                ret_dict = self.render_frame(frame_id, camera_id)
                
                img_fine = (ret_dict['output'] * 255).clip(0, 255)

                ret_img = Image.fromarray(img_fine.astype(np.uint8))   
                
                ret_img.save(os.path.join(
                    self.base_exp_dir, 'validations_fine', str(self.iter_step) +'_' + str(camera_id) + "_" + str(frame_id) + '.png'
                ))    
                
                psnr, ssim, maeValue, mseValue, lpips = self.val_img_loss(
                    output_img = img_fine, gt_img = np.transpose(ret_dict['gt_img'] * 255, axes=[1,2,0]).astype(np.uint8), gt_mask = ret_dict['gt_mask']
                ) 
                
                psnrs.append(psnr)
                ssims.append(ssim)
                maes.append(maeValue)
                mses.append(mseValue)
                lpips_val.append(lpips)
        
        psnrs = np.array(psnrs).mean()
        ssims = np.array(ssims).mean()
        maes = np.array(maes).mean()
        mses = np.array(mses).mean()
        lpips_val = np.array(lpips_val).mean()

        loss_dict = {
            "psnrs": psnrs,
            "maes" : maes,
            "mses" : mses,
            "lpips": lpips_val
        }

        concat_loss_str = lambda d: ', '.join([f'{key}: {value} ' for key, value in d.items()])
        
        print('iter step: ', self.iter_step,' ',concat_loss_str(loss_dict))
        
        if self.summary_writer is not None:
            for each_key in loss_dict:
                self.summary_writer.add_scalar(each_key, loss_dict[each_key], self.iter_step)
                
        return
        
    def train(self):
        ic('start training')
        
        res_step = self.end_iter - self.iter_step
        woot_bar = tqdm(range(res_step))
    
        ###################################################################################################################
        #                                        Now we simplify the pretraining                                          #
        ###################################################################################################################     
        
        if self.iter_step >= self.first_phase_end:
            self.dataset.chosen_frame_id_list = self.org_train_list
            print('second phase', self.dataset.chosen_frame_id_list)
        else:
            self.dataset.chosen_frame_id_list = self.mock_train_list
            print('first phase', self.dataset.chosen_frame_id_list)


        train_dataloader = DataLoader(
            self.dataset, batch_size=1, 
            shuffle=True, 
            num_workers = self.worker_num,
            collate_fn=list
        ) 
        
        self.unet_appearance.train()
        self.unet_geometry.train()
        
        self.update_learning_rate()
        
        it = iter(train_dataloader)
        
        criterion = torch.nn.MSELoss()

        for iter_i in woot_bar:

            # for the pretraining stage, only for the first few frames
            if self.iter_step == self.first_phase_end:
                print('restart dataset')
                
                self.dataset.chosen_frame_id_list = self.org_train_list
                
                train_dataloader = DataLoader(
                    self.dataset, batch_size=1, 
                    shuffle=True, 
                    num_workers = self.worker_num,
                    collate_fn=list
                )
                
                it = iter(train_dataloader) 
            
            self.optimizer.zero_grad()
            self.update_learning_rate()
            
            ###################################################################################################################
            
            ret_dict = next(it)[0]
            
            camera_id = ret_dict['cur_camera_id']
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
                        
            ###################################################################################################################

            gt_img = ret_dict['ret_img']            
            gt_img = torch.FloatTensor(gt_img).to(self.device)
            
            gt_mask = ret_dict['ret_mask']
            gt_mask = torch.FloatTensor(gt_mask).to(self.device)
            
            bbox = ret_dict['ret_bbox']         

            ###################################################################################################################
            
            geo_feats = self.unet_geometry(network_motion_cond).squeeze(0)        
            app_feats = self.unet_appearance(network_motion_cond, frame_global_translation).squeeze(0)
             
            geo_feats = geo_feats[:,self.non_empty_idx, self.non_empty_idy].permute(1, 0).contiguous()
            app_feats = app_feats[:,self.non_empty_idx, self.non_empty_idy].permute(1, 0).contiguous()
            
            canonical_delta = geo_feats[:,8:]
            canonincal_pos_map = canonincal_pos_map + canonical_delta 
            transform_vec[:, :3, 3] = transform_vec[:, :3, 3] / 1000.
            
            pos_xyz = self.compute_deformed_template(canonincal_pos_map, transform_vec)

            ################################################################################################################### 
            # a rough initalization for the model to fit the surface
            # to ease the pain to fit the indiviual frames
            
            if self.iter_step <= self.first_phase_end:
                _delta = geo_feats[:,8:]
                _scaling = geo_feats[:, :3]
                _rotation = geo_feats[:, 3:7]
                _opacity = geo_feats[:, 7].unsqueeze(1)
                _features_dc = app_feats[:, :3].reshape(-1, 1, 3)
                _features_rest = app_feats[:, 3:48].reshape(-1, 15, 3) 
                
                ret_ref_gaussian = self.create_ref_gaussian(
                    ret_dict
                )
                
                features_dc_loss = criterion(_features_dc, ret_ref_gaussian._features_dc)
                features_rest_loss = criterion(_features_rest, ret_ref_gaussian._features_rest)          

                scaling_loss = criterion(_scaling, ret_ref_gaussian._scaling)
                rotation_loss = criterion(_rotation, ret_ref_gaussian._rotation)
                opacity_loss = criterion(_opacity, ret_ref_gaussian._opacity)
                delta_loss = criterion(_delta, torch.zeros_like(_delta))
      
                final_loss = features_dc_loss + features_rest_loss + scaling_loss + rotation_loss + opacity_loss * 0.1 + delta_loss

                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()

                post_fix_dict = {
                    'loss': final_loss.cpu().item(),
                    'dc': features_dc_loss.cpu().item(),
                    'rest': features_rest_loss.cpu().item(),
                    'rot': rotation_loss.cpu().item(),
                    'op': opacity_loss.cpu().item(),
                    'del': delta_loss.cpu().item(),
                    'sc': scaling_loss.cpu().item()
                }
                
                woot_bar.set_postfix(post_fix_dict)
            
            else:
                ################################################################################################################### 
                # The real training      
                                 
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
                    image_height = self.dataset.img_height,
                    image_width = self.dataset.img_width
                )
                
                camera_info.world_view_transform = camera_info.world_view_transform.to(self.device)
                camera_info.projection_matrix = camera_info.projection_matrix.to(self.device)
                camera_info.full_proj_transform = camera_info.full_proj_transform.to(self.device)
                camera_info.camera_center = camera_info.camera_center.to(self.device)
                
                render_pkg = render(camera_info, self.gaussians, self.mocked_pipeline, self.background)
                
                image = render_pkg["render"]
                
                
                color_loss = l1_loss(image, gt_img)
                ssim_loss = ssim(image, gt_img)
                
                final_loss = self.color_weight * color_loss + self.ssim_weight * (1.0 - ssim_loss)
                
                self.optimizer.zero_grad()
                final_loss.backward()
                self.optimizer.step()

                post_fix_dict = {
                    'loss': final_loss.cpu().item(),
                    'fid':cur_frame_id,
                    'color': color_loss.cpu().item(),
                    'ssim': ssim_loss.cpu().item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                }

                woot_bar.set_postfix(post_fix_dict)
   
            if (self.iter_step % 10 == 0):
                self.profile_loss(post_fix_dict)

            if (self.iter_step % self.val_freq == 0):
                self.validate_image()
                
            if (self.iter_step % self.plot_histogram_freq == 0):
                self.plot_weight_histogram()

            if (self.iter_step % self.save_freq == 0):
                self.save_checkpoint()

            self.iter_step = self.iter_step + 1
                
 
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
        preload_conf, is_continue=True
    )
