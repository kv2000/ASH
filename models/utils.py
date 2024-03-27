"""
@File: utils.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2024-3-1
@Desc: Trimmed version for the dataset utils.
"""

import numpy as np

def gen_uv_barycentric(face_idx, face_texture_coords, resolution=512):
    uv_face_id = np.ones(shape=(resolution,resolution)) * (-1.0)
    uv_bary_weights = np.zeros(shape=(resolution,resolution,3))

    for i in range(face_idx.shape[0]):
        cur_face_uv_coords = face_texture_coords[i]
        uu_min = np.clip(np.min(cur_face_uv_coords[:,0]) * resolution - 2, 0, resolution - 1)
        uu_max = np.clip(np.max(cur_face_uv_coords[:,0]) * resolution + 2, 0, resolution - 1)
        vv_min = np.clip(np.min(cur_face_uv_coords[:,1]) * resolution - 2, 0, resolution - 1)
        vv_max = np.clip(np.max(cur_face_uv_coords[:,1]) * resolution + 2, 0, resolution - 1)
        uu_min, uu_max, vv_min, vv_max = int(uu_min), int(uu_max), int(vv_min), int(vv_max)

        for xx in range(uu_min, uu_max + 1):
            for yy in range(vv_min, vv_max + 1):
                fin_x, fin_y = xx, yy
                if uv_face_id[fin_x,fin_y] == -1:
                    px, py = (xx)/resolution, (yy)/resolution
                    
                    p0x, p0y = cur_face_uv_coords[0, 0], cur_face_uv_coords[0, 1]
                    p1x, p1y = cur_face_uv_coords[1, 0], cur_face_uv_coords[1, 1]
                    p2x, p2y = cur_face_uv_coords[2, 0], cur_face_uv_coords[2, 1]
                    
                    signed_area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)

                    w_1 = 1 / (2 * signed_area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
                    w_2 = 1 / (2 * signed_area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)
                    w_0 = 1 - w_1 - w_2

                    if (w_0 >= 0) and (w_1 >= 0) and (w_2 >= 0):
                        uv_face_id[fin_x, fin_y] = i
                        uv_bary_weights[fin_x, fin_y, 0] = w_0
                        uv_bary_weights[fin_x, fin_y, 1] = w_1
                        uv_bary_weights[fin_x, fin_y, 2] = w_2
                else:
                    pass

    uv_face_id = uv_face_id.astype(np.int32)

    return uv_face_id, uv_bary_weights