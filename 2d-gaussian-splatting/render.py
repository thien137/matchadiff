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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        
        # extract the mesh and save
        name = 'fuse_unbounded.ply'
        verts, faces, vert_colors = gaussExtractor.extract_mesh_unbounded2(resolution=args.mesh_res)
        device = 'cuda'
        
        from matcha.dm_scene.meshes import Meshes, TexturesVertex
        p3d_mesh = Meshes(
            verts=[verts.float().to(device)], 
            faces=[faces.long().to(device)],
            textures=TexturesVertex([vert_colors.float().to(device)]),
        )
        
        scene_cameras = scene.getTrainCameras()
        from matcha.dm_scene.cameras import CamerasWrapper, GSCamera
        gs_cameras = []
        for scene_camera in scene_cameras:
            gs_cameras.append(GSCamera(
                colmap_id=scene_camera.colmap_id,
                R=scene_camera.R,
                T=scene_camera.T,
                FoVx=scene_camera.FoVx,
                FoVy=scene_camera.FoVy,
                image=scene_camera.original_image,
                gt_alpha_mask=scene_camera.gt_alpha_mask,
                image_name=scene_camera.image_name,
                uid=scene_camera.uid,
                data_device=scene_camera.data_device,
                image_height=scene_camera.image_height,
                image_width=scene_camera.image_width,
            ))
        cameras_wrapper = CamerasWrapper(gs_cameras)
        from matcha.dm_scene.meshes import render_mesh_with_pytorch3d
        
        optimizer = torch.optim.Adam([verts], lr=1e-3)
        steps = 10
        for step in tqdm(range(steps)):
            for i in range(len(scene_cameras)):            
                result = render_mesh_with_pytorch3d(p3d_mesh, cameras_wrapper, i)
                result_image = result['rgb']
                gt_image = scene_cameras[i].original_image.permute(1, 2, 0)
                loss = torch.mean((result_image - gt_image) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Step {step}, Loss: {loss.item()}")
            # if step % 5 == 0:
            #     with torch.no_grad():
            #         result = render_mesh_with_pytorch3d(p3d_mesh, cameras_wrapper, 0)
            #         # write rgb images
            #         aname = f"zzzzz{step}" + name
            #         print("image saved at {}".format(os.path.join(train_dir, aname.replace('.ply', '_rgb.png'))))
            #         rgb_image = (result['rgb'].detach().cpu().numpy() * 255).astype(np.uint8)
            #         from PIL import Image
            #         Image.fromarray(rgb_image).save(os.path.join(train_dir, aname.replace('.ply', '_rgb.png')))
                    #Image.fromarray((scene_cameras[0].original_image.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)).save(os.path.join(train_dir, "original_" + aname.replace('.ply', '_original.png')))
        
        # o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        # print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # # post-process the mesh and save, saving the largest N clusters
        # mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        # o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        # print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))