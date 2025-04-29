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

import numpy as np
import torch
import trimesh
from skimage import measure
# modified from here https://github.com/autonomousvision/sdfstudio/blob/370902a10dbef08cb3fe4391bd3ed1e227b5c165/nerfstudio/utils/marching_cubes.py#L201
def marching_cubes_with_contraction(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
):
    assert resolution % 512 == 0

    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = torch.linspace(x_min, x_max, cropN).cuda()
                y = torch.linspace(y_min, y_max, cropN).cuda()
                z = torch.linspace(z_min, z_max, cropN).cuda()

                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                @torch.no_grad()
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 256**3, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3)
                points = points.reshape(-1, 3)
                pts_sdf = evaluate(points.contiguous())
                z = pts_sdf.detach().cpu().numpy()
                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                    )
                    verts = verts + np.array([x_min, y_min, z_min])
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)
                
                print("finished one block")

    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices(digits_vertex=6)

    # inverse contraction and clipping the points range
    if inv_contraction is not None:
        combined.vertices = inv_contraction(torch.from_numpy(combined.vertices).float().cuda()).cpu().numpy()
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)
    
    return combined

def marching_cubes_diffmc(
    sdf,
    bounding_box_min,
    bounding_box_max,
    resolution,
    level=0.0,
    device='cuda'
):
    """
    Differentiable marching cubes (no contraction version).
    
    Args:
        sdf (Tensor): [N, N, N] SDF volume.
        bounding_box_min (Tensor): [3] lower corner.
        bounding_box_max (Tensor): [3] upper corner.
        resolution (int): Number of voxels per axis (e.g., 256).
        level (float): Iso-surface level to extract (default 0.0).

    Returns:
        verts (Tensor): [V, 3] mesh vertices.
        faces (Tensor): [F, 3] mesh faces.
    """
    from diso import DiffDMC
    dmc = DiffDMC(dtype=torch.float32).to(device)

    print("Running differentiable marching cubes...")

    verts, faces = dmc(sdf.unsqueeze(0).unsqueeze(0), isovalue=level)  # Make [1,1,N,N,N]

    if verts.numel() == 0 or faces.numel() == 0:
        return torch.empty((0, 3), device=device), torch.empty((0, 3), dtype=torch.long, device=device)

    # Bring verts back to world coordinates
    scaling = (bounding_box_max - bounding_box_min) / (resolution - 1)
    verts = verts.squeeze(0)  # Remove batch dim
    faces = faces.squeeze(0)

    verts = verts * (resolution - 1)  # [0, 1] -> [0, resolution - 1]
    verts = verts * scaling + bounding_box_min  # Scale and translate

    return verts, faces

def marching_cubes_with_contraction_diffdmc(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0.0,
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
    device='cuda',
):
    assert resolution % 512 == 0

    resN = resolution
    cropN = 512
    N = resN // cropN

    xs = np.linspace(bounding_box_min[0], bounding_box_max[0], N + 1)
    ys = np.linspace(bounding_box_min[1], bounding_box_max[1], N + 1)
    zs = np.linspace(bounding_box_min[2], bounding_box_max[2], N + 1)

    from diso import DiffDMC
    dmc = DiffDMC(dtype=torch.float32).to(device)
    
    all_verts = []
    all_faces = []
    vert_offset = 0

    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(f"Processing block ({i}, {j}, {k})")

                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                # Build 3D grid
                x = torch.linspace(x_min, x_max, cropN, device=device)
                y = torch.linspace(y_min, y_max, cropN, device=device)
                z = torch.linspace(z_min, z_max, cropN, device=device)
                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                grid_points = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

                # Evaluate SDF at grid points
                sdf_vals = []
                for chunk in torch.split(grid_points, 1024**2):
                    sdf_vals.append(sdf(chunk))
                sdf_grid = torch.cat(sdf_vals, dim=0).view(cropN, cropN, cropN)

                # Run differentiable MC
                verts, faces = dmc(sdf_grid, isovalue=level)

                if verts.numel() == 0 or faces.numel() == 0:
                    continue

                # Transform normalized verts to world coordinates
                scaling = torch.tensor([
                    (x_max - x_min) / (cropN - 1),
                    (y_max - y_min) / (cropN - 1),
                    (z_max - z_min) / (cropN - 1),
                ], device=verts.device, dtype=verts.dtype)

                origin = torch.tensor([x_min, y_min, z_min], device=verts.device, dtype=verts.dtype)

                verts = verts * (cropN - 1)  # bring [0, 1] -> [0, cropN - 1]
                verts = verts * scaling + origin

                # Apply inverse contraction if needed
                if inv_contraction is not None:
                    verts = inv_contraction(verts)
                    verts = torch.clamp(verts, -max_range, max_range)

                # Offset face indices for merging
                faces = faces + vert_offset
                vert_offset += verts.shape[0]

                all_verts.append(verts)
                all_faces.append(faces)

                print("Finished block")

    final_verts = torch.cat(all_verts, dim=0)
    final_faces = torch.cat(all_faces, dim=0)

    if return_mesh:
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=final_verts.detach().cpu().numpy(),
            faces=final_faces.detach().cpu().numpy(),
            process=False,
        )
        return mesh
    
    return final_verts, final_faces