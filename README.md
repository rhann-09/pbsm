# Physics-Based SMPL-X Modelling (PBSM)

> [!WARNING]  
> This package is currently in development. All is subject to change without notice.

* Currently provides a tool to convert SMPL-X models into MuJoCo-compatible MJCF formats consisting of segmented convex-hull collision meshes and linear-blended skinning of the SMPLX-model mesh.


## Get the SMPL-X Models

To obtain the SMPL-X Models and uv mappings go to the SMPL-X website downloads section: https://smpl-x.is.tue.mpg.de/

## Example Usage

```python
import time
import torch
import numpy as np
import mujoco
import mujoco.viewer
import pbsm.mujoco_smplx.utils as utils
from pbsm.mujoco_smplx.main import smplx2mjcf

model_path = "/path/to/model.pkl"
gender = "female"
ext = "pkl"
betas = torch.tensor([[-1, -1, 1, 0, 1, -1, 1, 1, 1, 1]], dtype=torch.float32)
plotting = False
save = True
runtime = 300
subdivision_iterations = 0 
output_file = "smplx_full_body.xml"
obj_path = "/path/to/smplx_uv.obj"
texture_file = "/path/to/smplx_uv.png"

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
```

## Example with face expression movements
```python
import math
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import pbsm.mujoco_smplx.utils as utils
from pbsm.mujoco_smplx.main import smplx2mjcf

model_path = "/path/to/model.pkl"
gender = "female"
ext = "pkl"
betas = torch.tensor([[-1, -1, 1, 0, 1, -1, 1, 1, 1, 1]], dtype=torch.float32)
plotting = False
save = True
runtime = 300
subdivision_iterations = 0 
output_file = "smplx_full_body.xml"
obj_path = "/path/to/smplx_uv.obj"
texture_file = "/path/to/smplx_uv.png"

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


# Base Data for Delta Calculation
# Extract the neutral vertices so we can calculate the deltas later
base_output = smplx_model(return_verts=True)
base_verts = base_output.vertices.detach().cpu().numpy().squeeze()

# Create the rotational transform that matches our MJCF generation (Z-up conversion)
r = R.from_euler('xyz', (90, 0, 0), degrees=True)

# Initialize MuJoCo
model = mujoco.MjModel.from_xml_path(output_file)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Disable gravity so the model floats perfectly still for inspection
model.opt.gravity[:] = [0, 0, 0]

# Cache the original base skin vertices from MuJoCo memory
base_skin_vert = np.array(model.skin_vert).copy()

# Launch the passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    
    start_time = time.time()
    
    # Run the simulation loop
    while viewer.is_running() and data.time < runtime:
        step_start = time.time()

        # DYNAMIC VERTEX INJECTION (Facial Expressions)
        
        # Animate Expression over time using sine waves
        expr = torch.zeros([1, 10], dtype=torch.float32)
        expr[0, 0] = math.sin(data.time * 4.0) * 3.0  # Dynamic smile/frown
        expr[0, 1] = math.cos(data.time * 4.0) * 2.0  # Eyebrow movement
        
        # Forward pass through SMPL-X to get morphed vertices
        new_output = smplx_model(return_verts=True, expression=expr)
        new_verts = new_output.vertices.detach().cpu().numpy().squeeze()
        
        # Calculate the vertex deltas
        delta_verts = new_verts - base_verts
        
        # Rotate deltas to match MuJoCo's Z-up coordinate system
        rotated_delta = r.apply(delta_verts)
        
        # Inject back into MuJoCo's skin rest-pose
        # Flattening is required because model.skin_vertpos is a flat 1D C-array
        model.skin_vert[:] = base_skin_vert + rotated_delta

        # FORWARD KINEMATICS (Pose Override) 
        
        # Lock all hinge joints (index 7 onwards) to 0.0 radians
        data.qpos[7:] = 0.0
        
        # Zero out all joint velocities so the physics engine doesn't build up 
        # momentum and "fight" our positional override
        data.qvel[:] = 0.0

        # Advance time
        mujoco.mj_step(model, data)

        # Sync the viewer with the current physics state
        viewer.sync()

        # Crude real-time synchronization
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
```