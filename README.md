# Physics-Based SMPL-X Modelling (PBSM)

> [!WARNING]  
> This package is currently in development. All is subject to change without notice.

* Currently provides a tool to convert SMPL-X models into MuJoCo-compatible MJCF formats consisting of segmented convex-hull collision meshes and linear-blended skinning of the SMPLX-model mesh.

Example Usage:
```python

import sys
import torch
import mujoco
import mujoco.viewer
import time
import pbsm.mujoco_smplx.utils as utils
from pbsm.mujoco_smplx.main import smplx2mjcf

model_path = "path/to/model"
gender = "female"
ext = "pkl"
betas = torch.tensor([[-1, -1, 1, 0, 1, -1, 1, 1, 1, 1]], dtype=torch.float32)
plotting = False
save = True
stl_folder = "STL"
density_kg_per_m3 = 1000
subdivision_iterations = 2
output_file = "smplx_full_body.xml"
runtime = 300

# create smplx.SMPLX model
smplx_model = utils.default_smplx_model(model_path, gender, ext, betas)

# create mjcf file and stl meshes
smplx2mjcf(smplx_model, plotting,  save, stl_folder, density_kg_per_m3, subdivision_iterations)

# import mjcf file into Mujoco
model = mujoco.MjModel.from_xml_path(output_file)
data = mujoco.MjData(model)

# Reset physics state
mujoco.mj_resetData(model, data)

# Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set the joint visualization option in the viewer's scene
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    
    start_time = time.time()
    
    # Run the simulation loop
    while viewer.is_running() and data.time < runtime:
        step_start = time.time()

        # Advance physics
        mujoco.mj_step(model, data)

        # Sync the viewer with the current physics state
        viewer.sync()

        # Crude real-time synchronization to prevent the sim from running too fast
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)