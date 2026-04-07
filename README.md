# Physics-Based SMPL-X Modelling (PBSM)

> [!WARNING]  
> This package is currently in development. All is subject to change without notice.

* Provides a tool to convert SMPL-X models into MuJoCo-compatible MJCF formats consisting of segmented convex-hull collision meshes and linear-blended skinning of the SMPLX-model mesh.
* Provides a tool to convert .vrm files from vroid studio into MuJoCo-compatible MJCF formats where physics is modelled in MuJoCo and rendered using Three.js in the browser

## Get the SMPL-X Models

To obtain the SMPL-X Models and uv mappings go to the SMPL-X website downloads section: https://smpl-x.is.tue.mpg.de/

# TODO:

1. Update and check docstrings
2. Create example using smplx to control vrm with physics active
3. Control vrm expressions/hands using smplx
4. Implement smplx motion diffusion model to control vrm (https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1)
5. Automatic creation of index.html without broswer caching and automatic opening
6. Control other aspects of index.html from python during setup e.g. add in objectss, lighting terrain etc...