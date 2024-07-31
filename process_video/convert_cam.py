# the nice tool to convert the calibration file to json file provided by @Haokai gjgj

import os
import numpy as np
import math
import json

scale_to_meters = True  
invert_extr = True  

IMAGE_W = 1285
IMAGE_H = 940

INTR = []
EXTR = []


calibration_dir = ''
OUTPATH_Train = ''

with open(calibration_dir, 'r') as fp:
    while True:
        text = fp.readline()
        if not text:
            break

        elif "extrinsic" in text:

            extrinsic = np.array(text.split()[1:]).astype(np.float32).reshape((4, 4))

            if scale_to_meters:
                extrinsic[:3, 3] /= 1000.0
            if invert_extr:
                EXTR.append(np.linalg.inv(extrinsic))
            else:
                EXTR.append(extrinsic.reshape((4, 4)))

        elif "intrinsic" in text:

            intrinsic = np.array(text.split()[1:]).astype(np.float32).reshape((4, 4))[:3, :3]
            INTR.append(intrinsic)


out = dict()

w = float(IMAGE_W)
h = float(IMAGE_H)
k1 = float(0.)
k2 = float(0.)
p1 = float(0.)
p2 = float(0.)
aabb_scale = 16


frames = list()
for id in range(len(EXTR)):
    current_frame = dict()

    imagePath = "{:06}".format(id)
    current_frame.update({"file_path": imagePath})
    fl_x = float(INTR[id][0][0])
    fl_y = float(INTR[id][1][1])
    cx = float(INTR[id][0][2])
    cy = float(INTR[id][1][2])

    camera_angle_x = math.atan(float(w) / (float(fl_x) * 2)) * 2
    camera_angle_y = math.atan(float(h) / (float(fl_y) * 2)) * 2

    current_frame.update({"camera_angle_x": camera_angle_x})
    current_frame.update({"camera_angle_y": camera_angle_y})
    current_frame.update({"fl_x": fl_x})
    current_frame.update({"fl_y": fl_y})
    current_frame.update({"cx": cx})
    current_frame.update({"cy": cy})

    transform_matrix = EXTR[id]
    transform_matrix = transform_matrix[[2, 0, 1, 3], :]

    current_frame.update({"transform_matrix": transform_matrix})

    frames.append(current_frame)
out.update({"frames": frames})


for f in out["frames"]:
    f["transform_matrix"] = f["transform_matrix"].tolist()

frames = out["frames"]
train_cam_infos = [c for idx, c in enumerate(frames)]

out_train = dict()

out_train.update({"k1": k1})
out_train.update({"k2": k2})
out_train.update({"p1": p1})
out_train.update({"p2": p2})
out_train.update({"w": w})
out_train.update({"h": h})
out_train.update({"aabb_scale": aabb_scale})
out_train.update({"frames": train_cam_infos})

with open(OUTPATH_Train, "w") as f:
    json.dump(out_train, f, indent=4)
    