#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.camera_utils import Camera
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_scaling_rotation_inverse
import matplotlib.pyplot as plt


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


class GaussianModel:
    def __init__(self, sh_degree: int, fea_dim=0, with_motion_mask=True, **kwargs):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._label = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.with_motion_mask = with_motion_mask
        if self.with_motion_mask:
            # Masks stored as features
            fea_dim += 1
        self.fea_dim = fea_dim
        self.feature = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def param_names(self):
        return ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', '_label', 'max_radii2D', 'xyz_gradient_accum']

    @classmethod
    def build_from(cls, gs, **kwargs):
        new_gs = GaussianModel(**kwargs)
        new_gs._xyz = nn.Parameter(gs._xyz)
        new_gs._features_dc = nn.Parameter(torch.zeros_like(gs._features_dc))
        new_gs._features_rest = nn.Parameter(torch.zeros_like(gs._features_rest))       
        new_gs._scaling = nn.Parameter(gs._scaling)
        new_gs._rotation = nn.Parameter(gs._rotation)
        new_gs._opacity = nn.Parameter(gs._opacity)
        new_gs.feature = nn.Parameter(gs.feature)
        new_gs.max_radii2D = torch.zeros((new_gs.get_xyz.shape[0]), device="cuda")

        if hasattr(gs, "_label") and gs._label is not None:
            new_gs._label = gs._label.clone()
        else:
            new_gs._label = torch.full((new_gs.get_xyz.shape[0],), -1, dtype=torch.long, device="cuda")
        return new_gs

    @property
    def motion_mask(self):
        if self.with_motion_mask:
            return torch.sigmoid(self.feature[..., -1:])
        else:
            return torch.ones_like(self._xyz[..., :1])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.
        return self.rotation_activation(self._rotation + rotation_bias)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_label(self):
        return self._label

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1, d_rotation=None, gs_rot_bias=None):
        if d_rotation is not None:
            rotation = quaternion_multiply(self._rotation, d_rotation)
        else:
            rotation = self._rotation
        if gs_rot_bias is not None:
            rotation = rotation / rotation.norm(dim=-1, keepdim=True)
            rotation = quaternion_multiply(gs_rot_bias, rotation)
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)
    
    def get_covariance_inv(self):
        L = build_scaling_rotation_inverse(self.get_scaling, self._rotation)
        actual_covariance_inv = L @ L.transpose(1, 2)
        return actual_covariance_inv

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float=5., print_info=True, max_point_num=150_000):
        self.spatial_lr_scale = 5
        if type(pcd.points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        else:
            fused_point_cloud = pcd.points
        if type(pcd.colors) == np.ndarray:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = pcd.colors
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if print_info:
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.feature = nn.Parameter(-1e-2 * torch.ones([self._xyz.shape[0], self.fea_dim], dtype=torch.float32).to("cuda:0"), requires_grad=True)
        if self.with_motion_mask:
            self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])

        if hasattr(pcd, 'labels') and pcd.labels is not None:
            if isinstance(pcd.labels, np.ndarray):
                labels = torch.tensor(pcd.labels, dtype=torch.long, device="cuda")
            else:
                labels = pcd.labels.to("cuda")
        else:
            labels = torch.full((self._xyz.shape[0],), -1, dtype=torch.long, device="cuda")  # 默认 label 为 -1（未标注）

        self._label = labels

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.fea_dim >0:
            l.append(
                {'params': [self.feature], 'lr': training_args.feature_lr, 'name': 'feature'}
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale, lr_final=training_args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self.fea_dim):
            l.append('fea_{}'.format(i))
        l.append('label')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        
        if self.fea_dim > 0:
            feature = self.feature.detach().cpu().numpy()
            attributes = np.concatenate((attributes, feature), axis=1)
        
        if hasattr(self, '_label') and self._label is not None:
            label = self._label.detach().cpu().numpy().reshape(-1, 1)
            attributes = np.concatenate((attributes, label), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def save_ply(self, path, use_label_color=False):
    #     mkdir_p(os.path.dirname(path))

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)

    #     if use_label_color and hasattr(self, '_label') and self._label is not None:
    #         import pdb
    #         pdb.set_trace()
    #         # overwrite f_dc with label colors
    #         label = self._label.detach().cpu()
    #         f_dc = torch.zeros((label.shape[0], 3), dtype=torch.float32)

    #         color_map = {
    #             1: torch.tensor([1.0, 0.0, 0.0]),  # Red
    #             2: torch.tensor([0.0, 1.0, 0.0]),  # Green
    #             3: torch.tensor([0.0, 0.0, 1.0])   # Blue
    #         }

    #         for class_id, color in color_map.items():
    #             f_dc[label == class_id] = color

    #         f_dc = f_dc.numpy()
    #         f_dc = f_dc.reshape(-1, 3)  # (N, 3)
    #     else:
    #         # use original feature color
    #         f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)

    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

    #     if self.fea_dim > 0:
    #         feature = self.feature.detach().cpu().numpy()
    #         attributes = np.concatenate((attributes, feature), axis=1)

    #     if hasattr(self, '_label') and self._label is not None:
    #         label = self._label.detach().cpu().numpy().reshape(-1, 1)
    #         attributes = np.concatenate((attributes, label), axis=1)

    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
        feas = np.zeros((xyz.shape[0], self.fea_dim))
        for idx, attr_name in enumerate(fea_names):
            feas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if "label" in plydata.elements[0].data.dtype.names:
            labels = np.asarray(plydata.elements[0]["label"]).astype(np.int64)
            self._label = torch.tensor(labels, dtype=torch.long, device="cuda")
        else:
            self._label = torch.zeros(xyz.shape[0], dtype=torch.long, device="cuda")

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.fea_dim > 0:
            self.feature = nn.Parameter(torch.tensor(feas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.fea_dim > 0:
            self.feature = optimizable_tensors["feature"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if hasattr(self, "_label"):
            self._label = self._label[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        
        if hasattr(self, "_label") and "label" in tensors_dict:
            label_tensor = tensors_dict["label"].to(self._label.device).view(-1)
            self._label = torch.cat([self._label, label_tensor], dim=0)

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_feature=None, new_label=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        
        if self.fea_dim > 0:
            d["feature"] = new_feature

        if hasattr(self, "_label") and new_label is not None:
            d["label"] = new_label

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.fea_dim > 0:
            self.feature = optimizable_tensors["feature"]

        if hasattr(self, "_label") and "label" in optimizable_tensors:
            self._label = optimizable_tensors["label"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads=None, grad_threshold=None, scene_extent=None, N=2, selected_pts_mask=None, without_prune=False):
        if selected_pts_mask is None:
            n_init_points = self.get_xyz.shape[0]
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_feature = self.feature[selected_pts_mask].repeat(N, 1) if self.fea_dim > 0 else None

        new_label = self._label[selected_pts_mask].repeat(N, 1) if hasattr(self, "_label") else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_feature, new_label)

        if not without_prune:
            prune_filter = torch.cat(
                (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)

    def densify_and_clone(self, grads=None, grad_threshold=None, scene_extent=None, selected_pts_mask=None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling,
                                                            dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_feature = self.feature[selected_pts_mask] if self.fea_dim > 0  else None

        new_label = self._label[selected_pts_mask] if hasattr(self, "_label") else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_feature, new_label)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def project_xyz_to_image(self, xyz_world: torch.Tensor, camera: Camera):
        """
        Project 3D points to 2D image coordinates using simple matrix math.

        Args:
            xyz_world: (N, 3) tensor of points in world space.
            camera: Camera object containing intrinsics and full_proj_transform.

        Returns:
            - uv: (N, 2) image pixel coordinates (u, v)
            - depth: (N,) depth in camera clip space (used for filtering)
        """
        N = xyz_world.shape[0]
        xyz_h = torch.cat([xyz_world, torch.ones((N, 1), device=xyz_world.device)], dim=1)  # (N, 4)

        # Apply projection transformation
        proj_pts = xyz_h @ camera.full_proj_transform  # (N, 4)
        proj_pts = proj_pts[:, :2] / proj_pts[:, 3:]  # perspective divide
        
        # Convert NDC to pixel coordinates - try different scaling
        # The issue might be that we need to center the coordinates properly
        proj_pts = (proj_pts + 1) / 2 * torch.tensor([camera.image_width, camera.image_height], device=xyz_world.device)
        
        # Try to center the coordinates - subtract offset to bring them into proper range
        proj_pts[:, 0] = proj_pts[:, 0] - (proj_pts[:, 0].mean() - camera.image_width / 2)
        proj_pts[:, 1] = proj_pts[:, 1] - (proj_pts[:, 1].mean() - camera.image_height / 2)

        # Calculate actual distances from camera to each Gaussian
        camera_pos = torch.tensor([camera.camera_center[0], camera.camera_center[1], camera.camera_center[2]], device=xyz_world.device)
        distances = torch.norm(xyz_world - camera_pos, dim=1)  # (N,) - actual distances

        return proj_pts, distances

    def render_clean_image(self, camera: Camera, save_path: str = None):
        """
        Render a clean image of 3D Gaussians without background for segmentation.
        Also returns the Gaussian-to-pixel mapping for label assignment.
        
        Args:
            camera: Camera object for rendering
            save_path: Optional path to save the rendered image
            
        Returns:
            rendered_image: Clean rendered image tensor (3, H, W)
            gaussian_to_pixel: Dict containing 'proj_pts', 'depth', and 'image_size'
        """
        from gaussian_renderer import render
        import torchvision
        import os
        
        # Create a simple pipeline for rendering
        class SimplePipe:
            def __init__(self):
                self.debug = False
                self.compute_cov3D_python = False
                self.convert_SHs_python = False
        
        pipe = SimplePipe()
        # Use white background for clean segmentation
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        
        # Render with no deformation (original 3D positions)
        with torch.no_grad():
            results = render(camera, self, pipe, bg_color, 
                           d_xyz=torch.zeros_like(self._xyz), 
                           d_rotation=torch.zeros((self._xyz.shape[0], 4), device=self._xyz.device),
                           d_scaling=torch.zeros_like(self._xyz))
            
            # Get the rendered image (RGB channels only)
            rendered_image = results["render"]  # (3, H, W)
            
            # Get the actual projection data from the render pipeline
            H, W = rendered_image.shape[1], rendered_image.shape[2]
            
            # The render pipeline should provide viewspace_points
            # if "viewspace_points" in results and results["viewspace_points"] is not None:
            #     viewspace_points = results["viewspace_points"]
            #     # viewspace_points should be in NDC coordinates, convert to pixel coordinates
            #     proj_pts = viewspace_points[:, :2]  # (N, 2) - [u, v] in NDC
            #     depth = viewspace_points[:, 2]  # (N,) - depth
                
            #     # Convert from NDC to pixel coordinates
            #     proj_pts[:, 0] = (proj_pts[:, 0] + 1) * 0.5 * W  # u: [-1,1] -> [0,W]
            #     proj_pts[:, 1] = (proj_pts[:, 1] + 1) * 0.5 * H  # v: [-1,1] -> [0,H]
            # else:
            # Fallback to manual projection if render doesn't provide viewspace_points
            proj_pts, distances = self.project_xyz_to_image(self._xyz, camera)
            
            # Create Gaussian-to-pixel mapping
            gaussian_to_pixel = {
                'proj_pts': proj_pts,  # (N, 2) - 2D projections
                'depth': distances,  # (N,) - actual distances from camera
                'image_size': (H, W)
            }
            
            # Save if path provided
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torchvision.utils.save_image(rendered_image, save_path)
                print(f"Clean rendered image saved to: {save_path}")
            
        return rendered_image, gaussian_to_pixel

    def assign_label_from_gaussian_to_pixel_mapping(self, mask_tensor: torch.Tensor, gaussian_to_pixel: dict, label_vote_count=None, vis_save_path: str = None):
        """
        Assign labels to Gaussians using the pre-computed Gaussian-to-pixel mapping.
        Optionally visualize projected points over segmentation mask and save to disk.
        
        Args:
            mask_tensor: Segmentation mask tensor (H, W)
            gaussian_to_pixel: Dict containing 'proj_pts', 'depth', and 'image_size'
            label_vote_count: Existing vote count tensor for majority voting
            vis_save_path: Optional path to save visualization image
            
        Returns:
            label_vote_count: Updated vote count tensor
        """
        device = self._xyz.device
        H, W = mask_tensor.shape

        # Get projection points and depth
        proj_pts = gaussian_to_pixel['proj_pts']  # (N, 2)
        depth = gaussian_to_pixel['depth']  # (N,)
        
        # Convert to integer pixel coordinates
        u_int = proj_pts[:, 0].round().long()
        v_int = proj_pts[:, 1].round().long()
        
        # Check bounds and valid depth (distances should be positive)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (depth > 0)
        vote_indices = torch.where(valid)[0]
        label_values = mask_tensor[v_int[valid], u_int[valid]]

        if len(label_values) == 0:
            return label_vote_count
        
        # Remove background (label 0)
        nonzero = label_values != 0
        vote_indices = vote_indices[nonzero]
        label_values = label_values[nonzero]

        # Convert label_values to long dtype for indexing
        label_values = label_values.long()
        
        if len(vote_indices) == 0:
            return label_vote_count
        
        # Apply distance-based filtering for each ball class
        filtered_vote_indices = []
        filtered_label_values = []
        
        # Get the maximum label number from the mask
        max_label = int(mask_tensor.max().item())
        
        for label_id in range(1, max_label + 1):  # Ball classes start from 1
            label_mask = (label_values == label_id)
            if label_mask.any():
                # Get distances for Gaussians with this label
                label_vote_indices = vote_indices[label_mask]
                label_depths = depth[label_vote_indices]
                
                if len(label_depths) > 0:
                    min_distance = label_depths.min()
                    max_distance = label_depths.max()
                    distance_range = max_distance - min_distance
                    
                    # Calculate adaptive margin: 0.35 * distance range
                    adaptive_margin = 0.3 * distance_range
                    
                    # Keep only Gaussians within adaptive distance margin
                    close_mask = label_depths <= (min_distance + adaptive_margin)
                    
                    filtered_vote_indices.append(label_vote_indices[close_mask])
                    filtered_label_values.append(label_values[label_mask][close_mask])
        
        # Combine filtered results
        if filtered_vote_indices:
            vote_indices = torch.cat(filtered_vote_indices)
            label_values = torch.cat(filtered_label_values)
        else:
            return label_vote_count
        
        num_points = self._xyz.shape[0]
        max_class_id = int(mask_tensor.max().item())

        if label_vote_count is None:
            label_vote_count = torch.zeros((num_points, max_class_id + 1), dtype=torch.long, device=device)

        # Check bounds before indexing
        if vote_indices.max() >= num_points:
            valid_votes = vote_indices < num_points
            vote_indices = vote_indices[valid_votes]
            label_values = label_values[valid_votes]
        
        if label_values.max() >= max_class_id + 1:
            valid_labels = label_values < max_class_id + 1
            vote_indices = vote_indices[valid_labels]
            label_values = label_values[valid_labels]
        
        # Update vote count
        if len(vote_indices) > 0:
            label_vote_count[vote_indices, label_values] += 1

        # Note: Depth-based outlier detection will be applied globally after all cameras
        
        # ========= Visualization part =========
        if vis_save_path is not None:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for server environments

            mask_np = mask_tensor.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(mask_np, cmap='gray')

            # Convert all projected points to CPU for visualization
            u_all = proj_pts[:, 0].detach().cpu().numpy()
            v_all = proj_pts[:, 1].detach().cpu().numpy()
            
            # Show all projected Gaussian points in light gray
            ax.scatter(u_all, v_all, s=0.5, color='lightgray', alpha=0.3, label='All projected Gaussians')
            
            # Show points that lie within 2D segmentation balls with colors
            colors = ['red', 'green', 'blue', 'yellow', 'purple']
            for label_id in range(1, max_class_id + 1):
                select = (label_values == label_id)
                if select.any():
                    ax.scatter(
                        proj_pts[vote_indices[select], 0].detach().cpu().numpy(),
                        proj_pts[vote_indices[select], 1].detach().cpu().numpy(),
                        s=2,
                        label=f'Label {label_id} (in segmentation)',
                        color=colors[(label_id - 1) % len(colors)],
                        alpha=0.8
                    )
            
            ax.set_title("All Projected 3D Gaussians vs Segmentation Mask (from Gaussian-to-Pixel Mapping)")
            ax.legend()
            ax.axis('off')
            
            # Ensure debug directory exists
            debug_dir = os.path.dirname(vis_save_path)
            if debug_dir:  # Only create if there's a directory path
                os.makedirs(debug_dir, exist_ok=True)
            else:  # If no directory, create 'debug' folder
                os.makedirs("debug", exist_ok=True)
                vis_save_path = os.path.join("debug", os.path.basename(vis_save_path))
            
            plt.savefig(vis_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Visualization saved to: {vis_save_path}")

        return label_vote_count

    def create_labeled_object_manager(self, control_gaussians: 'GaussianModel'):
        """
        Create a labeled object manager that maps control points and Gaussians with the same label.
        This allows applying 6DOF transformations to entire objects (control points + Gaussians).
        
        Args:
            control_gaussians: GaussianModel instance for control points
            
        Returns:
            LabeledObjectManager instance
        """
        return LabeledObjectManager(control_gaussians, self)

    def assign_label_from_mask_majority(self, mask_tensor: torch.Tensor, camera: Camera, label_vote_count=None, vis_save_path: str = None):
        """
        Use 2D mask to assign semantic labels to Gaussians via majority voting.
        Optionally visualize projected points over segmentation mask and save to disk.
        """
        device = self._xyz.device
        H, W = mask_tensor.shape

        # Project Gaussians to image space
        proj_uv, depth = self.project_xyz_to_image(self._xyz, camera)
        u, v = proj_uv[:, 0], proj_uv[:, 1]

        # Round to nearest integer
        u_int = u.round().long()
        v_int = v.round().long()

        # Mask valid points
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (depth > 0)
        vote_indices = torch.where(valid)[0]
        label_values = mask_tensor[v_int[valid], u_int[valid]]

        # Remove background (label 0)
        nonzero = label_values != 0
        vote_indices = vote_indices[nonzero]
        label_values = label_values[nonzero]

        num_points = self._xyz.shape[0]
        max_class_id = int(mask_tensor.max().item())

        if label_vote_count is None:
            label_vote_count = torch.zeros((num_points, max_class_id + 1), dtype=torch.long, device=device)

        label_vote_count[vote_indices, label_values] += 1

        # ========= Visualization =========
        if vis_save_path is not None:
            import matplotlib
            matplotlib.use('Agg')  

            mask_np = mask_tensor.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(mask_np, cmap='gray')

            # Convert all projected points to CPU for visualization
            u_all = u.detach().cpu().numpy()
            v_all = v.detach().cpu().numpy()
            
            # Show all projected Gaussian points in light gray
            ax.scatter(u_all, v_all, s=0.5, color='lightgray', alpha=0.3, label='All projected Gaussians')
            
            # Show points that lie within 2D segmentation balls with colors
            colors = ['red', 'green', 'blue', 'yellow', 'purple']
            for label_id in range(1, max_class_id + 1):
                select = (label_values == label_id)
                if select.any():
                    ax.scatter(
                        u[valid][nonzero][select].detach().cpu().numpy(),
                        v[valid][nonzero][select].detach().cpu().numpy(),
                        s=2,
                        label=f'Label {label_id} (in segmentation)',
                        color=colors[(label_id - 1) % len(colors)],
                        alpha=0.8
                    )

            ax.set_title("All Projected 3D Gaussians vs Segmentation Mask")
            ax.legend()
            ax.axis('off')
            os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
            plt.savefig(vis_save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        return label_vote_count

    def color_gaussians_by_label(self):
        """
        Use _label to color _features_dc as visual indicator:
        label 1 → red, 2 → green, 3 → blue
        """
        if not hasattr(self, "_label") or self._label is None:
            print("[Warning] No labels found on gaussians. Skipping coloring.")
            return

        with torch.no_grad():
            f_dc = self._features_dc  # shape: (N, 1, 3)
            labels = self._label      # shape: (N,)

            label_color_map = {
                1: torch.tensor([1.0, 0.0, 0.0], device=f_dc.device),  # Red
                2: torch.tensor([0.0, 1.0, 0.0], device=f_dc.device),  # Green
                3: torch.tensor([0.0, 0.0, 1.0], device=f_dc.device),  # Blue
            }

            for class_id, color in label_color_map.items():
                mask = (labels == class_id)
                if mask.any():
                    f_dc[mask] = color.view(1, 1, 3).expand_as(f_dc[mask])

class StandardGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, fea_dim=0, with_motion_mask=True, all_the_same=False):
        super().__init__(sh_degree, fea_dim, with_motion_mask)
        self.all_the_same = all_the_same
    
    @property
    def get_scaling(self):
        scaling = self._scaling.mean()[None, None].expand_as(self._scaling) if self.all_the_same else self._scaling.mean(dim=1, keepdim=True).expand_as(self._scaling)
        return self.scaling_activation(scaling)


class LabeledObject(nn.Module):
    """
    Represents a single labeled object containing control points and Gaussians with the same label.
    This class wraps both control points and Gaussians, sharing the same coordinate space.
    """
    
    def __init__(self, label: int, control_gaussians: 'GaussianModel', gaussians: 'GaussianModel', 
                 control_indices: torch.Tensor, gaussian_indices: torch.Tensor):
        """
        Args:
            label: The label ID for this object
            control_gaussians: Reference to the control Gaussian model
            gaussians: Reference to the scene Gaussian model
            control_indices: Indices of control points belonging to this object
            gaussian_indices: Indices of Gaussians belonging to this object
        """
        super().__init__()
        
        self.label = label
        self.control_gaussians = control_gaussians
        self.gaussians = gaussians
        self.control_indices = control_indices
        self.gaussian_indices = gaussian_indices
        
        # Calculate object center (average of all points)
        control_centers = control_gaussians._xyz[control_indices].mean(dim=0)
        gaussian_centers = gaussians._xyz[gaussian_indices].mean(dim=0)
        self.center = (control_centers + gaussian_centers) / 2
        
        print(f"Created LabeledObject {label}: {len(control_indices)} control points, {len(gaussian_indices)} Gaussians")
    
    def get_control_xyz(self) -> torch.Tensor:
        """Get positions of control points for this object."""
        return self.control_gaussians._xyz[self.control_indices]
    
    def get_gaussian_xyz(self) -> torch.Tensor:
        """Get positions of Gaussians for this object."""
        return self.gaussians._xyz[self.gaussian_indices]
    
    def set_control_xyz(self, xyz: torch.Tensor):
        """Set positions of control points for this object."""
        self.control_gaussians._xyz.data[self.control_indices] = xyz
    
    def set_gaussian_xyz(self, xyz: torch.Tensor):
        """Set positions of Gaussians for this object."""
        self.gaussians._xyz.data[self.gaussian_indices] = xyz
    
    def get_info(self) -> dict:
        """Get information about this object."""
        return {
            'label': self.label,
            'num_control_points': len(self.control_indices),
            'num_gaussians': len(self.gaussian_indices),
            'center': self.center.detach().cpu().numpy()
        }


class LabeledObjectManager(nn.Module):
    """
    Manages multiple LabeledObject instances.
    Each labeled object is a separate class instance containing control points and Gaussians with the same label.
    """
    
    def __init__(self, control_gaussians: 'GaussianModel', gaussians: 'GaussianModel'):
        """
        Args:
            control_gaussians: GaussianModel instance for control points
            gaussians: GaussianModel instance for scene Gaussians
        """
        super().__init__()
        
        # Get labels from both models
        control_labels = control_gaussians._label
        gaussian_labels = gaussians._label
        
        # Get unique labels (excluding background label 0)
        unique_labels = torch.unique(torch.cat([control_labels, gaussian_labels]))
        unique_labels = unique_labels[unique_labels > 0]
        
        print(f"Found {len(unique_labels)} labeled objects: {unique_labels.tolist()}")
        
        # Create a LabeledObject instance for each label
        self.objects = nn.ModuleDict()
        
        for label in unique_labels:
            label_item = label.item()
            
            # Find control points and Gaussians with this label
            control_mask = (control_labels == label)
            gaussian_mask = (gaussian_labels == label)
            
            control_indices = torch.where(control_mask)[0]
            gaussian_indices = torch.where(gaussian_mask)[0]
            
            if len(control_indices) > 0 and len(gaussian_indices) > 0:
                # Create a LabeledObject for this label
                labeled_obj = LabeledObject(
                    label_item, control_gaussians, gaussians,
                    control_indices, gaussian_indices
                )
                self.objects[str(label_item)] = labeled_obj
    
    def get_object(self, label: int) -> LabeledObject:
        """Get the LabeledObject instance for a specific label."""
        label_key = str(label)
        if label_key in self.objects:
            return self.objects[label_key]
        else:
            print(f"Warning: Object with label {label} not found")
            return None
    
    def apply_transformation(self, label: int):
        """Apply transformation to a specific object."""
        obj = self.get_object(label)
        if obj is not None:
            obj.apply_transformation()
    
    def apply_all_transformations(self):
        """Apply transformations to all objects."""
        for obj in self.objects.values():
            obj.apply_transformation()
    
    def set_transformation(self, label: int, translation: torch.Tensor, rotation_quat: torch.Tensor):
        """Set transformation for a specific object and apply it."""
        obj = self.get_object(label)
        if obj is not None:
            obj.translation.data = translation
            obj.rotation.data = rotation_quat
            obj.apply_transformation()
    
    def reset_transformation(self, label: int):
        """Reset transformation for a specific object."""
        obj = self.get_object(label)
        if obj is not None:
            obj.reset_transformation()
    
    def reset_all_transformations(self):
        """Reset all transformations to identity."""
        for obj in self.objects.values():
            obj.reset_transformation()
    
    def get_object_info(self) -> dict:
        """Get information about all objects."""
        return {int(label): obj.get_info() for label, obj in self.objects.items()}
    
    def get_all_labels(self) -> list:
        """Get all object labels."""
        return [int(label) for label in self.objects.keys()]
