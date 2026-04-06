#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: Sun Mar 29 14:02:31 2026
Author: Reuben H. (mango)

Project:
    Physics-Based SMPL-X Modelling (PBSM)
"""
import time
import torch
import numpy as np
import mujoco
import mujoco.viewer
import pbsm.mujoco_smplx.utils as utils
from pbsm.main import smplx2mjcf


model_path = "/home/mango/pbsm_dev/models/SMPLX_FEMALE.pkl"
gender = "female"
ext = "pkl"
betas = torch.tensor([[-1, -1, 1, 0, 1, -1, 1, 1, 1, 1]], dtype=torch.float32)
plotting = False
save = True
runtime = 300
subdivision_iterations = 0 
output_file = "smplx_full_body.xml"
obj_path = "/home/mango/pbsm_dev/models/smplx_uv.obj"
texture_file = "/home/mango/pbsm_dev/models/smplx_uv.png"

# Build Model
smplx_model = utils.default_smplx_model(model_path, gender, ext, betas)

# make mjcf file 
smplx2mjcf(smplx_model,
           plotting=plotting,
           save=save,
           subdivision_iterations=subdivision_iterations,
           obj_path=obj_path,
           texture_file=texture_file,
           output_file=output_file)

# Initialize MuJoCo
model = mujoco.MjModel.from_xml_path(output_file)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Cache the original base skin vertices from MuJoCo memory
base_skin_vert = np.array(model.skin_vert).copy()

# Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    
    start_time = time.time()
    
    # Run the simulation loop
    while viewer.is_running() and data.time < runtime:
        step_start = time.time()

        # Advance time
        mujoco.mj_step(model, data)

        # Sync the viewer with the current physics state
        viewer.sync()

        # Crude real-time synchronization
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)