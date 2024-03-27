"""
@File: dataset.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2024-3-1
@Desc: The dataset adpated for testing ASH on DynaCap dataset.
"""

import sys
import os
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from models.utils import gen_uv_barycentric
import DeepCharacters_Pytorch.CSVHelper as CSVHelper
from DeepCharacters_Pytorch.character_utils import dual_quad_to_trans_vec
# the verison for ASH to expose the quats
from DeepCharacters_Pytorch.WootCharacter_ASH import WootCharacterWithQuat as WootCharacter
from DeepCharacters_Pytorch.WootGCN import WootSpatialGCN

from kornia.geometry.conversions import quaternion_to_rotation_matrix, Rt_to_matrix4x4
from kornia.geometry.quaternion import QuaternionCoeffOrder

import pickle as pkl
import random
import math
import cv2 as cv
import trimesh

class ASH_Inference_Dataset(Dataset):
    def __init__(self, conf, device = 'cuda'):
        print("+++++ start creating dataset ")
        self.conf = conf
        self.never_stop_size = int(1e7)
        self.device = device
        
        # captry skeleton pose is accepted
        self.dof_dir = self.conf['dataset.skeleton_angles']
        self.dof_angle_normalized_dir = self.conf['dataset.skeleton_angles_rotation_normalized']
                
        self.val_frame_start = self.conf.get_int('dataset.val_frame_start')
        self.val_frame_end = self.conf.get_int('dataset.val_frame_end')
        self.val_sample_interval = self.conf.get_int('dataset.val_sample_interval')

        self.val_frame_idx = [
            i for i in range(self.val_frame_start, self.val_frame_end, self.val_sample_interval)
        ]
        
        print("+++++ valiation frames :", self.val_frame_idx, len(self.val_frame_idx))
        
        #################################################################################################
        
        self.charactor = None
        self.initialize_charactor()
        
        #################################################################################################
        
        self.spatial_gcn = None
        self.initialize_spatial_gcn()

        self.delta_gcn = None
        self.delta_checkpoint_dir = self.conf['train.delta_checkpoint_dir']
        
        self.initialize_delta_gcn()
        self.is_load_delta_checkpoints = self.conf.get_bool('train.load_delta_checkpoints', default=False)
        
        if self.is_load_delta_checkpoints:
            self.load_delta_init_checkpoint()

        self.spatial_gcn.eval()
        self.delta_gcn.eval()
        
        for each_param in self.spatial_gcn.parameters():
            each_param.grad = None

        for each_param in self.delta_gcn.parameters():
            each_param.grad = None      

        #################################################################################################

        self.dof_arr = None
        self.dof_angle_normalized_arr = None
        
        self.num_dof = -1
        self.tot_frame_num = -1

        self.load_dof()
        
        #################################################################################################     
        
        self.camera_num = self.conf.get_int('dataset.camera_num')
        self.val_camera = self.conf.get_list('dataset.val_camera')
        
        #################################################################################################         
        
        self.uv_face_id = None
        self.uv_bary_weights = None
        self.uv_non_empty = None
        self.uv_non_empty_mask = None
        self.uv_coord_img = None
        
        self.uv_idx_np = None
        self.uv_idx_cu = None
        self.uv_idy_np = None
        self.uv_idy_cu = None

        self.uv_vert_idx_np = None
        self.uv_vert_idx_cu = None
        self.uv_bary_weights_cu = None

        #################################################################################################      
        self.obj_reader = None
        
        self.face_idx = None
        self.face_texture_coords = None
        self.vert_adj_faces = None
        self.vert_adj_weights = None

        self.face_idx_cu = None
        self.vert_adj_faces_cu = None
        self.vert_adj_weights_cu = None

        self.vert_num = None

        self.max_vert_ind = 0
        self.min_vert_ind = 11451419
        self.barycentric_tex_size = self.conf.get_int('dataset.barycentric_tex_size')

        self.init_template_mesh_info()

        self.real_global_translation_dir = self.conf['model.extra_settings.real_global_translation_dir']
        self.real_global_translation = np.array([0., -0.5, 0.])
        self.real_global_scale = 2.5
        self.load_global_translation()
       
        print("+++++ end creating dataset ")
        
        return
    
    def init_template_mesh_info(self):
        print('+++++ create template mesh related from:', self.charactor.template_mesh_dir)
        
        self.obj_reader = self.charactor.obj_reader
        
        self.face_idx = np.array(self.obj_reader.facesVertexId).reshape([-1, 3])
        self.face_texture_coords = np.array(self.obj_reader.textureCoordinates).reshape([-1,3,2])
        self.vert_num = self.obj_reader.numberOfVertices
        
        self.gen_adjacent_list()        
        
        self.gen_barycentric_coords()
        
        print('+++++ end create template mesh related')
        return 

    def load_dof(self):
        print("+++++ Loading All Sorts of dofs")
        max_end_frame = self.val_frame_end + 1

        self.dof_arr = CSVHelper.load_csv_sequence_2D(
            self.dof_dir, type='float', skipRows=1, skipColumns=1, end_frame=(max_end_frame + 1)
        )
        self.num_dof = self.dof_arr.shape[-1]
        self.tot_frame_num = self.dof_arr.shape[0]

        self.dof_angle_normalized_arr = (CSVHelper.load_csv_compact_4D(
            self.dof_angle_normalized_dir, 3, self.num_dof, 1, 1, 1, 'float', end_frame=(3 * (max_end_frame + 1))
        )).reshape((-1, 3, self.num_dof))
        
        print(
            ' dof shape: ', self.dof_arr.shape, '\n',
            ' dof rotation normalized shape: ',self.dof_angle_normalized_arr.shape
        ) 
        print("+++++ Finished Loading All Sorts of dofs")
        return

    def load_delta_init_checkpoint(self):
        print('+++++ init with delta checkpoint', self.delta_checkpoint_dir)
        
        if os.path.isfile(self.delta_checkpoint_dir):
            cur_state_dict = torch.load(self.delta_checkpoint_dir, map_location=self.device)      
            if (self.spatial_gcn is not None) and ('spatial_gcn' in cur_state_dict.keys()):
                print('+++++ loading checkpoints spatial_gcn')
                self.spatial_gcn.load_state_dict(cur_state_dict['spatial_gcn'])     
            
            if (self.delta_gcn is not None) and ('delta_gcn' in cur_state_dict.keys()):
                print('+++++ loading checkpoints delta_gcn')
                self.delta_gcn.load_state_dict(cur_state_dict['delta_gcn'])             
        else:
            print(self.delta_checkpoint_dir, 'check point not found')
        
        return 

    def initialize_charactor(self):
        print('+++++ start initializing the character ')
        
        self.charactor = WootCharacter(
            **self.conf['character'],
            device=self.device
        )
                
        print('+++++ end initializing the character ')
        return

    def initialize_spatial_gcn(self):
        print('+++++ start initializing the spatial gcn ')

        self.spatial_gcn = WootSpatialGCN(
            **self.conf['spatial_gcn'],
            obj_reader=self.charactor.graph_obj_reader, device=self.device
        )
        self.spatial_gcn = self.spatial_gcn.to(self.device)

        print('+++++ end initializing the spatial gcn ')
        return 
    
    def initialize_delta_gcn(self):
        print('+++++ start initializing the delta gcn ')
        
        self.delta_gcn = WootSpatialGCN(
            **self.conf['delta_gcn'],
            obj_reader=self.charactor.obj_reader, device=self.device
        )
        self.delta_gcn = self.delta_gcn.to(self.device)
        print('+++++ end initializing the delta gcn ')
        return 

    def gen_adjacent_list(self):   
        
        self.vert_adj_faces = []
        self.vert_adj_weights = []
        
        temp_adj_list = [[] for i in range(self.vert_num)]

        for i in range(self.face_idx.shape[0]):
            t0, t1, t2 = self.face_idx[i][0], self.face_idx[i][1], self.face_idx[i][2]
            temp_adj_list[t0].append(i)
            temp_adj_list[t1].append(i)
            temp_adj_list[t2].append(i)
        
        for i in range(len(temp_adj_list)):
            self.max_vert_ind = max(len(temp_adj_list[i]), self.max_vert_ind)
            self.min_vert_ind = min(len(temp_adj_list[i]), self.min_vert_ind)
        
        assert self.min_vert_ind >= 1
        
        for i in range(len(temp_adj_list)):
            cur_adj_num = len(temp_adj_list[i])
            tmp_faces_idx = []
            tmp_weights = []
            for j in range(self.max_vert_ind + 1):
                if j < cur_adj_num:
                    tmp_faces_idx.append(temp_adj_list[i][j])
                    tmp_weights.append(1.0/cur_adj_num)
                else:
                    tmp_faces_idx.append(temp_adj_list[i][-1])
                    tmp_weights.append(0.0)
            
            self.vert_adj_faces.append(tmp_faces_idx)
            self.vert_adj_weights.append(tmp_weights)
        
        self.vert_adj_faces = np.array(self.vert_adj_faces, dtype=np.int32)
        self.vert_adj_weights = np.array(self.vert_adj_weights, dtype=np.float32)

        self.face_idx_cu = torch.LongTensor(self.face_idx).to(self.device)
        self.vert_adj_faces_cu = torch.LongTensor(self.vert_adj_faces).to(self.device)
        self.vert_adj_weights_cu = torch.FloatTensor(self.vert_adj_weights).to(self.device)
        
        return     
    
    def gen_barycentric_coords(self):
        print('+++++ create mesh barycentric related')

        self.uv_face_id, self.uv_bary_weights = gen_uv_barycentric(
            self.face_idx, self.face_texture_coords, resolution=self.barycentric_tex_size, 
        )
        self.uv_idx_np, self.uv_idy_np = np.where(self.uv_face_id >= 0)
        
        self.uv_idx_cu = torch.LongTensor(self.uv_idx_np).to(self.device)
        self.uv_idy_cu = torch.LongTensor(self.uv_idy_np).to(self.device)

        self.uv_vert_idx_np = self.face_idx[self.uv_face_id[self.uv_idx_np, self.uv_idy_np],:]
        self.uv_vert_idx_cu = torch.LongTensor(self.uv_vert_idx_np).to(self.device)
        
        self.uv_bary_ind = np.argmax(self.uv_bary_weights, axis = -1)
        self.uv_bary_ind_cu = torch.LongTensor(self.uv_bary_ind[self.uv_idx_np, self.uv_idy_np]).to(self.device)
        self.uv_bary_iden_cu = torch.LongTensor([i for i in range(self.uv_bary_ind_cu.shape[0])]).to(self.device)
        
        self.uv_bary_weights_np = self.uv_bary_weights[self.uv_idx_np, self.uv_idy_np,:]
        self.uv_bary_weights_cu = torch.FloatTensor(self.uv_bary_weights_np).to(self.device)

        self.uv_non_empty = np.where(self.uv_face_id >= 0)
        self.uv_non_empty_mask = self.uv_face_id >= 0 

        self.gen_uv_map()

        print('+++++ end create mesh barycentric related')
        return 

    def gen_uv_map(self):
        print('+++++ create mesh uv maps')
        self.uv_coord_img = np.zeros([
            self.uv_non_empty_mask.shape[0], 
            self.uv_non_empty_mask.shape[1],
            2
        ])

        normalized_face_tex = (self.face_texture_coords - 0.5) * 2.0

        w = self.uv_bary_weights[self.uv_non_empty[0], self.uv_non_empty[1],:]

        f_id = self.uv_face_id[self.uv_non_empty[0], self.uv_non_empty[1]]
        
        p = normalized_face_tex[f_id]

        w_p = p[:,0,:] * w[:,0:1] + p[:,1,:] * w[:,1:2] + p[:,2,:] * w[:,2:3]

        self.uv_coord_img[self.uv_non_empty[0], self.uv_non_empty[1],:2] = w_p
        
        print('+++++ end create mesh uv map')
        return 
    
    def compute_vert_and_face_normal(self, verts):

        v0 = verts[:,self.face_idx_cu[:,0]]
        v1 = verts[:,self.face_idx_cu[:,1]]
        v2 = verts[:,self.face_idx_cu[:,2]]
        
        v2v1 = v2 - v1
        v0v1 = v0 - v1 

        temp_face_normal = torch.cross(v2v1, v0v1, dim = -1)
        temp_face_normal = F.normalize(temp_face_normal, dim = -1)

        temp_vert_normal = temp_face_normal[:,self.vert_adj_faces_cu]
        temp_vert_weights = self.vert_adj_weights_cu[None,...,None]
        
        fin_vert_normal = torch.sum(temp_vert_normal * temp_vert_weights, dim=2)     
        
        fin_vert_normal = F.normalize(fin_vert_normal, dim = -1)
            
        return fin_vert_normal, temp_face_normal 

    def render_feature_tex(self, feats, resolution):
        feat_dim = feats.shape[-1]
        batch_size = feats.shape[0]
        
        ret_feat = torch.zeros([batch_size, resolution, resolution, feat_dim], device=self.device)

        p = feats[:,self.uv_vert_idx_cu, :]
        w = self.uv_bary_weights_cu[None,...]

        weighted_feats = p[:,:,0,:] * w[:,:,0,None] + p[:,:,1,:] * w[:,:,1,None] + p[:,:,2,:] * w[:,:,2,None]

        ret_feat[:,self.uv_idx_cu, self.uv_idy_cu,:] = weighted_feats
        
        return ret_feat

    def render_quad_tex(self, feats_rot, feats_trans, resolution):
        feat_dim = feats_rot.shape[-1]
        
        p_rot = feats_rot[self.uv_vert_idx_cu, :]
        p_trans = feats_trans[self.uv_vert_idx_cu, :]
        
        w = self.uv_bary_weights_cu
        
        p_fst = p_rot[self.uv_bary_iden_cu, self.uv_bary_ind_cu,:]
        
        p_sign = (torch.sum(p_rot * p_fst[:,None,:], dim = -1) > 0).float()
                
        fin_sign = p_sign * 2 - 1
        
        weighted_feats_rot = p_rot[:,0,:] * fin_sign[:,0,None] * w[:,0,None] + p_rot[:,1,:] * fin_sign[:,1,None]  * w[:,1,None] + p_rot[:,2,:] * fin_sign[:,2,None] * w[:,2,None]   
        weighted_feats_trans = p_trans[:,0,:] * fin_sign[:,0,None] * w[:,0,None] + p_trans[:,1,:] * fin_sign[:,1,None]  * w[:,1,None] + p_trans[:,2,:] * fin_sign[:,2,None] * w[:,2,None] 
        
        raw_scale = torch.norm(
            weighted_feats_rot, p=2, dim = -1
        )
        
        scale_mask = (raw_scale < 1e-9).float()
        fin_scale = 1. / (raw_scale + scale_mask)

        weighted_feats_rot = weighted_feats_rot * fin_scale[...,None]
        weighted_feats_trans = weighted_feats_trans  * fin_scale[...,None]

        fin_rot_mat = quaternion_to_rotation_matrix(
            weighted_feats_rot.reshape([-1,4]), QuaternionCoeffOrder.WXYZ
        ).reshape([-1, 3, 3])       

        fin_trans_vec = dual_quad_to_trans_vec(
            weighted_feats_rot, weighted_feats_trans
        ).reshape([-1, 3])

        fin_transform_mat = Rt_to_matrix4x4(
            fin_rot_mat, fin_trans_vec[...,None]
        ).reshape([-1, 4, 4])   

        return fin_transform_mat

    def __len__(self):
        return max(len(self.chosen_frame_id_list), self.never_stop_size)

    def __getitem__(self, idx):
        
        ret_dict = {}
        
        return ret_dict

    def load_global_translation(self):
        
        f = open(self.real_global_translation_dir,'r')
        
        self.real_global_translation = np.array([float(x) for x in f.readlines()[0].split(' ')]).astype(np.float32)
        self.real_global_scale = 2.5
        
        f.close()

        print(
            self.real_global_scale,
            self.real_global_translation
        )
                
        return 

    def get_val_img_dict(self, frame_id):

        with torch.no_grad():
            
            cur_frame_id = frame_id
            history_frame_id = np.array([
                cur_frame_id, max(cur_frame_id - 1, 0), max(cur_frame_id - 2, 0)
            ])
                                
            anglesNormalized0 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,0,:]).to(self.device)
            anglesNormalized1 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,1,:]).to(self.device)
            anglesNormalized2 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,2,:]).to(self.device)
            
            concated_angles_normalized = torch.cat(
                [anglesNormalized0, anglesNormalized1, anglesNormalized2],  dim = 0
            )

            pose_only_template, picked_r, picked_t = self.charactor.compute_posed_template_embedded_graph(
                dof = concated_angles_normalized
            )
            
            v0 = pose_only_template[0:3, ...]
            v1 = pose_only_template[3:6, ...]
            v2 = pose_only_template[6:9, ...]

            inputTemporalPoseDeltaNet = torch.concat(
                [v0, v1, v2], dim = -1
            )

            r0 = picked_r[0:3, :, :]
            r1 = picked_r[3:6, :, :]
            r2 = picked_r[6:9, :, :]

            t0 = picked_t[0:3, :, :]
            t1 = picked_t[3:6 :, :]
            t2 = picked_t[6:9, :, :]

            inputTemporalPoseEGNet = torch.concat([
                t0 / 1000., r0, 
                t1 / 1000., r1, 
                t2 / 1000., r2
            ], dim = 2)

            eg_node_RT = self.spatial_gcn(inputTemporalPoseEGNet)
            delta_T = eg_node_RT[:, :, :3] * 1000.
            delta_R = eg_node_RT[:, :, 3:6]

            per_vertex_deformation = self.delta_gcn(inputTemporalPoseDeltaNet)
            per_vertex_deformation = per_vertex_deformation * 1000.0    

            dofs = self.dof_arr[history_frame_id,...]
            dofs = torch.FloatTensor(dofs.copy()).to(self.device)        
                        
            ret_posed_delta, delta_canoincal, fin_translation_quad, fin_rotation_quad, skeletal_joints = self.charactor.forward_test(
                dof = dofs, delta_R = delta_R, delta_T = delta_T, per_vertex_T = per_vertex_deformation
            )
                        
            skeletal_joints = skeletal_joints.detach().cpu().numpy()[0]

            cur_displacement = self.dof_arr[cur_frame_id][:3]
            
            root_normalized_ret_posed_delta = ret_posed_delta.clone() - torch.FloatTensor(cur_displacement[None, None, :]).to(self.device) * 1000.
            
            temp_vert_normal, temp_face_normal = self.compute_vert_and_face_normal(
                root_normalized_ret_posed_delta
            ) 
            
            concated_position_features = torch.cat([root_normalized_ret_posed_delta, delta_canoincal[:1,:,:3], temp_vert_normal], dim = 0)
            
            concated_feature_map = self.render_feature_tex(
                feats = concated_position_features,
                resolution = self.barycentric_tex_size
            ) 
            
            concated_feature_map = concated_feature_map.detach().cpu().numpy()
            
            ddc_cond_map = concated_feature_map[:3,:,:,:]
            ddc_cond_map = np.concatenate([ddc_cond_map[0], ddc_cond_map[1], ddc_cond_map[2]], axis = -1)
                        
            canoincal_pose_map = concated_feature_map[3:4,:,:,:]
            posed_normal_map = concated_feature_map[4:7,:,:,:]
            posed_normal_map = np.concatenate([posed_normal_map[0], posed_normal_map[1], posed_normal_map[2]], axis = -1)
            
            #########################################################################################################
            fin_translation_quad = fin_translation_quad[0]   
            fin_rotation_quad = fin_rotation_quad[0]
            fin_rotation_quad = torch.nn.functional.normalize(fin_rotation_quad, dim = -1)         

            fin_transform_vec = self.render_quad_tex(
                feats_rot = fin_rotation_quad,
                feats_trans = fin_translation_quad,
                resolution = self.barycentric_tex_size
            )
            
            fin_transform_vec = fin_transform_vec.detach().cpu().numpy()
        
        frame_global_translation = self.dof_arr[cur_frame_id,:3] / 1.2 - self.real_global_translation
                            
        ret_dict = {
            'cur_frame_id': cur_frame_id,
            'transform_vec': fin_transform_vec, 
            'ddc_cond_map': ddc_cond_map,
            'canoincal_pos_map': canoincal_pose_map[:,self.uv_idx_np, self.uv_idy_np, :], 
            'posed_normal_map': posed_normal_map,
            'frame_global_translation':frame_global_translation,
            'skeletal_joints':skeletal_joints
        }
        
        return ret_dict
