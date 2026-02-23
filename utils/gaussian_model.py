import os
import torch
import numpy as np
from utils.feature_utils import *
from plyfile import PlyData, PlyElement


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.prune_features = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
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
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc =torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def get_point_number(self):
        return self._xyz.shape[0]
    
    def get_prune_input_f15(self, knn_k, knn_method="ivf"):
        N = self.get_point_number()
        positions = self.get_xyz           # [N, 3]
        opacities = self.get_opacity       # [N, 1] activated
        scales = self.get_scaling          # [N, 3] activated
        shs = self.get_features

        shs_dc = self.get_features_dc  # [N, 1, 3]

        sh_view = shs_dc.transpose(1, 2).view(N, 3, 1)
        rgb = torch.clamp(C0 * sh_view[..., 0] + 0.5, 0.0, 1.0)

        xyz = positions.cpu().numpy()

        if knn_method in ["ivf", "brute_force"]:
            if not HAS_CUVS:
                raise ImportError(f"cuvs is required for '{knn_method}' method. Please install cuvs.")
            if knn_method == "ivf":
                n_lists = min(8192, max(100, N // 500))
                n_probes = min(16, max(1, n_lists // 512))
                # n_probes = min(32, max(4, n_lists // 128))
                knn_indices, knn_dists = knn_cuvs_ivf_flat(xyz, knn_k, n_lists=n_lists, n_probes=n_probes)
                knn_indices = torch.tensor(knn_indices, dtype=torch.long).cuda()
                knn_dists = fix_knn_dists(torch.tensor(knn_dists, dtype=torch.float32).cuda())
            elif knn_method == "brute_force":
                knn_indices, knn_dists = knn_cuvs_brute_force(xyz, knn_k)
                knn_indices = torch.tensor(knn_indices, dtype=torch.long).cuda()
                knn_dists = torch.tensor(knn_dists, dtype=torch.float32).cuda()
        elif knn_method == "ckdtree":
            if not HAS_SCIPY:
                raise ImportError(f"scipy is required for '{knn_method}' method. Please install scipy.")
            knn_indices, knn_dists = knn_ckdtree(xyz, knn_k)
            knn_indices = torch.tensor(knn_indices, dtype=torch.long).cuda()
            knn_dists = torch.tensor(knn_dists, dtype=torch.float32).cuda()
        else:
            raise ValueError(f"Unknown knn_method '{knn_method}'. Supported methods: 'ivf', 'brute_force', 'ckdtree'")
        
        f_p_avg_dists = knn_dists.mean(axis=1)
        f_p_avg_dists_z_score_local = compute_knn_z_score(f_p_avg_dists, knn_indices, pooling="none")
        f_p_avg_dists_z_score_global = z_score_tensor(torch.log(f_p_avg_dists), cutoff=0)
        

        rgb_z_score_local = compute_knn_z_score(rgb, knn_indices, pooling="mean")
        f_p_sh_anisotropy = compute_sh_anisotropy_loop_std(shs, num_dirs=60, max_sh_degree=self.max_sh_degree)
        f_p_sh_anisotropy_z_score_local = compute_knn_z_score(f_p_sh_anisotropy, knn_indices, pooling="none")
        f_p_sh_anisotropy_z_score_global = z_score_tensor(torch.log(f_p_sh_anisotropy), cutoff=0)

        f_p_s_sorted, _ = torch.sort(scales, dim=1)  # ascending order
        f_p_volume = torch.prod(scales, dim=1)
        
        f_p_s_sorted_z_score_local = compute_knn_z_score(f_p_s_sorted, knn_indices, pooling="none")
        f_p_s_sorted_z_score_global = z_score_tensor(torch.log(f_p_s_sorted), cutoff=0)
        
        f_p_volumn_z_score_local = compute_knn_z_score(f_p_volume, knn_indices, pooling="none")
        f_p_volumn_z_score_global = z_score_tensor(torch.log(f_p_volume), cutoff=0)
        
        f_p_opacity_z_score_local = compute_knn_z_score(opacities, knn_indices, pooling="none")
        f_p_opacity_z_score_global = z_score_tensor(torch.log(opacities), cutoff=0)
        
        input_features = torch.cat((f_p_avg_dists_z_score_local, 
                                    f_p_avg_dists_z_score_global, 
                                    f_p_sh_anisotropy_z_score_local,
                                    f_p_sh_anisotropy_z_score_global, 
                                    f_p_s_sorted_z_score_local, 
                                    f_p_s_sorted_z_score_global, 
                                    f_p_volumn_z_score_local, 
                                    f_p_volumn_z_score_global, 
                                    f_p_opacity_z_score_local, 
                                    f_p_opacity_z_score_global, 
                                    rgb_z_score_local), dim=1)
        prune_features = percentile_cutoff_normalize(input_features, lower_percentile=0.01, upper_percentile=0.99, cut_off_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        self.prune_features = prune_features
    
    def prune_points(self, valid_mask):
        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]
