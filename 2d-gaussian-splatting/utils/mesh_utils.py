#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()
        
        ### NEW ###
        self.compute_scene_bounds()

    def compute_scene_bounds(self):
        """
        Computes min and max 3D bounding box from the current Gaussians.
        """
        xyz = self.gaussians.get_xyz
        margin = 0.05 * (xyz.max(dim=0)[0] - xyz.min(dim=0)[0])  # 5% margin
        self.scene_bounds_min = xyz.min(dim=0)[0] - margin
        self.scene_bounds_max = xyz.max(dim=0)[0] + margin
        
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []

    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())

        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    def extract_mesh_bounded_differentiable(
        self,
        voxel_size=0.004,
        sdf_trunc=0.02,
        depth_trunc=3.0,
        mask_background=True
    ):
        """
        Differentiable version of extract_mesh_bounded.
        Returns verts, faces, colors.
        """
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm
        from diso import DiffDMC
        import numpy as np

        device = 'cuda'

        print("Running differentiable TSDF volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_trunc: {depth_trunc}')

        # === Helper: Project points and compute SDF + RGB
        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            new_points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1.) & (pix_coords < 1.) & (z > 0)).all(dim=-1)

            sampled_depth = F.grid_sample(
                depthmap[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True
            ).reshape(-1, 1)

            sampled_rgb = F.grid_sample(
                rgbmap[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True
            ).reshape(3, -1).T

            sdf = sampled_depth - z
            return sdf, sampled_rgb, mask_proj

        # === Helper: Marching cubes on bounded TSDF
        def marching_cubes_diffdmc_bounded(tsdf_grid, voxel_size, min_xyz, level=0.0, device='cuda'):
            Nx, Ny, Nz = tsdf_grid.shape
            dmc = DiffDMC(dtype=torch.float32).to(device)

            verts, faces = dmc(tsdf_grid, isovalue=level)

            if verts.numel() == 0 or faces.numel() == 0:
                print("[Warning] Marching cubes returned empty mesh.")
                return verts.new_zeros((0,3)), faces.new_zeros((0,3))

            scale_xyz = torch.tensor([Nx-1, Ny-1, Nz-1], device=verts.device)
            verts = verts * scale_xyz
            verts_world = min_xyz[None, :] + verts * voxel_size

            return verts_world, faces

        # === Compute bounding box
        def get_bounds(viewpoint_stack):
            all_positions = torch.stack([torch.tensor(view.T).float().to(device) for view in viewpoint_stack], dim=0)
            center = all_positions.mean(dim=0)
            extent = (all_positions.max(dim=0).values - all_positions.min(dim=0).values) * 1.5
            min_xyz = center - extent / 2
            max_xyz = center + extent / 2
            return min_xyz, max_xyz

        def compute_grid_size(bounds, voxel_size):
            min_xyz, max_xyz = bounds
            sizes = (max_xyz - min_xyz)
            grid_size = torch.ceil(sizes / voxel_size).long()
            return tuple(grid_size)

        # === Build voxel grid
        min_xyz, max_xyz = get_bounds(self.viewpoint_stack)
        Nx, Ny, Nz = compute_grid_size((min_xyz, max_xyz), voxel_size)
        print(f'Voxel grid size: {Nx} x {Ny} x {Nz}')

        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(Nx, device=device),
            torch.arange(Ny, device=device),
            torch.arange(Nz, device=device),
            indexing='ij'
        )
        voxels = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()
        voxels_world = min_xyz[None, None, None, :] + voxels * voxel_size
        voxels_flat = voxels_world.reshape(-1, 3)  # (N,3)

        # === Initialize TSDF and color
        tsdfs = torch.ones((voxels_flat.shape[0],), device=device)
        rgbs = torch.zeros((voxels_flat.shape[0], 3), device=device)
        weights = torch.ones_like(tsdfs)

        # === Integrate from all views
        for i, viewpoint_cam in enumerate(tqdm(self.viewpoint_stack, desc="TSDF integration progress")):
            rgb = self.rgbmaps[i].to(device)
            depth = self.depthmaps[i].to(device)

            depth = depth.clone()
            depth = depth.clamp(0.0, depth_trunc)

            if mask_background and (viewpoint_cam.gt_alpha_mask is not None):
                depth[viewpoint_cam.gt_alpha_mask < 0.5] = 0.0

            sdf, rgb_sampled, mask_proj = compute_sdf_perframe(
                i, voxels_flat, depthmap=depth, rgbmap=rgb, viewpoint_cam=viewpoint_cam
            )

            sdf = sdf.flatten()

            mask_proj = mask_proj & (sdf > -sdf_trunc)

            if mask_proj.sum() == 0:
                continue

            sdf_clamped = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]

            w = weights[mask_proj]
            wp = w + 1

            tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf_clamped) / wp
            rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb_sampled[mask_proj]) / wp[:, None]
            weights[mask_proj] = wp

        # === Reshape TSDF grid
        tsdf_grid = tsdfs.view(Nx, Ny, Nz)
        color_grid = rgbs.view(Nx, Ny, Nz, 3)

        # === Run marching cubes
        verts_world, faces = marching_cubes_diffdmc_bounded(
            tsdf_grid=tsdf_grid,
            voxel_size=voxel_size,
            min_xyz=min_xyz,
            level=0.0,
            device=device
        )

        if verts_world.numel() == 0 or faces.numel() == 0:
            return verts_world, faces, verts_world.new_zeros((0, 3))

        # === Sample vertex colors
        verts_idx = (verts_world - min_xyz[None, :]) / voxel_size
        verts_idx = verts_idx.long()
        verts_idx[:, 0] = verts_idx[:, 0].clamp(0, Nx-1)
        verts_idx[:, 1] = verts_idx[:, 1].clamp(0, Ny-1)
        verts_idx[:, 2] = verts_idx[:, 2].clamp(0, Nz-1)

        sampled_colors = color_grid[
            verts_idx[:,0],
            verts_idx[:,1],
            verts_idx[:,2]
        ]

        return verts_world, faces, sampled_colors
        
    def extract_mesh_unbounded_train_nvdiffrast(self, resolution=1024):
        """
        Returns verts, faces, and vertex_colors (all torch tensors) for use with NVDiffRast.
        """

        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)
            return torch.where(mag < 1, y, (1 / (2 - mag)) * (y / mag))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            new_points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1.) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(
                depthmap[None], pix_coords[None, None], mode='bilinear',
                padding_mode='border', align_corners=True
            ).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(
                rgbmap[None], pix_coords[None, None], mode='bilinear',
                padding_mode='border', align_corners=True
            ).reshape(3, -1).T
            sdf = sampled_depth - z
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1 / (2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:, 0])
            rgbs = torch.zeros((samples.shape[0], 3), device=samples.device)
            weights = torch.ones_like(samples[:, 0])

            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration"):
                sdf, rgb, mask_proj = compute_sdf_perframe(
                    i, samples,
                    depthmap=self.depthmaps[i].cuda(),
                    rgbmap=self.rgbmaps[i].cuda(),
                    viewpoint_cam=viewpoint_cam,
                )
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:, None] + rgb[mask_proj]) / wp[:, None]
                weights[mask_proj] = wp

            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        # Contract/normalize helpers
        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        # Settings
        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing SDF grid @ {N}³, voxel size: {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)

        # Compute bounding region
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = min(np.quantile(R, 0.95) + 0.01, 1.9)

        # Extract mesh via differentiable marching cubes
        from utils.mcube_utils import marching_cubes_with_contraction_diffdmc
        print("Extracting mesh...")
        verts, faces = marching_cubes_with_contraction_diffdmc(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
            return_mesh=False,
        )

        # Optionally compute RGBs
        print("Texturing mesh with RGBs...")
        _, rgbs = compute_unbounded_tsdf(verts, inv_contraction=None, voxel_size=voxel_size, return_rgb=True)

        # Final return — PyTorch-native
        return verts, faces, rgbs
    
    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction, marching_cubes_with_contraction_diffdmc
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction_diffdmc(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
            return_mesh=True,
        )

        def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
            # Create an Open3D TriangleMesh object
            o3d_mesh = o3d.geometry.TriangleMesh()

            # Set the vertices and triangles (faces)
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

            # Optionally add normals if they exist
            if mesh.vertex_normals is not None:
                o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals)

            return o3d_mesh

        mesh = trimesh_to_open3d(mesh)
        
        # coloring the mesh
        torch.cuda.empty_cache()
        #mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        print(rgbs)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
