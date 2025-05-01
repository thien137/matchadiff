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
    xs = torch.linspace(grid_min[0], grid_max[0], N + 1)
    ys = torch.linspace(grid_min[1], grid_max[1], N + 1)
    zs = torch.linspace(grid_min[2], grid_max[2], N + 1)

    meshes = []
    verts_all = []
    faces_all = []
    vert_offset = 0
    
    from diso import DiffDMC
    diffdmc = DiffDMC(dtype=torch.float32).to('cuda')

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

                from torch.utils.checkpoint import checkpoint

                def evaluate(points, chunk=256**3, use_checkpoint=True):
                    out = []

                    for pnts in torch.split(points, chunk, dim=0):
                        if use_checkpoint:
                            out.append(checkpoint(sdf, pnts))
                        else:
                            out.append(sdf(pnts))

                    return torch.cat(out, dim=0)

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3)
                points = points.reshape(-1, 3)
                pts_sdf = evaluate(points.contiguous())
                z = pts_sdf.reshape(cropN, cropN, cropN).float().cuda()
                if not (z.min() > level or z.max() < level):
                    verts, faces = diffdmc(z, normalize=False, return_quads=False)  # verts in [0, dim-1]

                    # Scale verts to world coordinates in this block
                    spacing = torch.tensor([
                        (x_max - x_min) / (cropN - 1),
                        (y_max - y_min) / (cropN - 1),
                        (z_max - z_min) / (cropN - 1)
                    ], device=verts.device)

                    origin = torch.tensor([x_min, y_min, z_min], device=verts.device)

                    verts = verts * spacing + origin
                     
                    faces = faces + vert_offset
                    verts_all.append(verts)
                    faces_all.append(faces)
                    vert_offset += verts.shape[0]
                    
                    print(verts.requires_grad)
                
                print("finished one block")

    verts_combined = torch.concatenate(verts_all, axis=0)
    faces_combined = torch.concatenate(faces_all, axis=0)

    def merge_close_vertices(verts, faces, tol=1e-6):
        # Quantize and hash vertices
        quantized = torch.round(verts / tol).to(torch.int64)  # (N, 3)
        hash_vals = (
            quantized[:, 0] * 73856093 +
            quantized[:, 1] * 19349663 +
            quantized[:, 2] * 83492791
        )

        # Get unique hashes and inverse mapping
        unique_hashes, inverse_indices = torch.unique(hash_vals, return_inverse=True)

        # Average the positions of all vertices assigned to the same hash (optional, or just take first)
        new_verts = torch.zeros((unique_hashes.shape[0], 3), device=verts.device, dtype=verts.dtype)
        counts = torch.bincount(inverse_indices)

        new_verts.index_add_(0, inverse_indices, verts)
        new_verts = new_verts / counts.unsqueeze(1)

        # Remap face indices
        new_faces = inverse_indices[faces]

        return new_verts, new_faces
    
    verts_combined, faces_combined = merge_close_vertices(verts_combined, faces_combined)

    if inv_contraction is not None:
        verts_combined = inv_contraction(verts_combined.cuda())
        verts_combined = torch.clip(verts_combined, -max_range, max_range)
        
    if return_mesh:
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=verts_combined.detach().cpu().numpy(),
            faces=faces_combined.detach().cpu().numpy(),
            process=False,
        )
        return mesh
    
    return verts_combined, faces_combined