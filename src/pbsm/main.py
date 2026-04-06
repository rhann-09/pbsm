#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Create MCJF file from SMPLX model.
"""

# Standard Imports
import os
import shutil

# Third-Party Imports
import trimesh
import smplx
import mujoco

# Package-Specific Imports
from pbsm.mujoco_smplx import utils as ms_utils
from pbsm.mujoco_smplx import plot as ms_plot
from pbsm.mujoco_vrm import utils as mv_utils


def smplx2mjcf(model: smplx.SMPLX,
               plotting: bool = False,
               save: bool = True,
               stl_folder: str = "STL",
               density_kg_per_m3: float = 1000.0,
               subdivision_iterations: int = 2,
               obj_path: str = None,
               num_vertices: int = 10475,
               texture_file: str = "smplx_texture.png",
               output_file: str = "smplx_full_body.xml",) -> None:

    """
    Converts an SMPL-X parametric body model into a MuJoCo compatible MJCF format.

    This function extracts the vertices, faces, joints, and Linear Blend Skinning (LBS) 
    weights from a provided SMPL-X model. It optionally subdivides the mesh for higher 
    resolution, enforces strict bilateral symmetry, segments the body parts, generates 
    convex hulls for physics collision, and writes the complete kinematic tree to an XML file.

    Parameters
    ----------
    model : smplx.SMPLX
        The instantiated SMPL-X model to convert.
    plotting : bool, optional
        If True, displays interactive Plotly 3D visualisations of the mesh, 
        vertices, segmented parts, and convex hulls during processing, by default False.
    save : bool, optional
        If True, saves the generated STL collision meshes and the final MJCF XML file 
        to the local disk, by default True.
    stl_folder : str, optional
        The name of the directory where the generated STL files for each body segment 
        will be saved, by default "STL".
    density_kg_per_m3 : float, optional
        The density applied to the generated convex hulls for physics calculations, 
        by default 1000.0.
    subdivision_iterations : int, optional
        The number of times to subdivide the mesh and interpolate weights for a 
        smoother surface. Set to 0 to skip subdivision, by default 2.
    num_vertices : int
        The number of vertices, by default 10475.
    texture_file : str
        Path to texture .png file, by default "smplx_texture.png".
    output_file : str, optional
        The filename for the generated MuJoCo XML (MJCF) file, by default "smplx_full_body.xml".

    Returns
    -------
    None
    """

    if isinstance(obj_path, str):
        uv_coords = ms_utils.load_aligned_smplx_uv(obj_path, num_vertices)
        uv_coords[:, 1] = 1.0 - uv_coords[:, 1]
    else:
        uv_coords = None
    
    stl_folder = os.path.join(os.getcwd(), stl_folder)
    
    segment_joints = ['jaw','head','neck','spine3','spine2','spine1','pelvis','right_collar','right_shoulder',
                      'right_elbow','right_wrist','right_index1','right_index2','right_index3','right_middle1',
                      'right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1',
                      'right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3','left_collar',
                      'left_shoulder','left_elbow','left_wrist','left_index1','left_index2','left_index3',
                      'left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3',
                      'left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3','left_hip',
                      'left_knee','left_ankle','left_foot','right_hip','right_knee','right_ankle','right_foot']
    
    names, network = ms_utils.make_name_and_network()
    
    if plotting:
        print("Plotting vertices and mesh")
        ms_plot.plot_mesh(model)
        ms_plot.plot_vertices(model)
    
    # Get Base Data
    output = model(return_verts=True)
    base_vertices = output.vertices.detach().cpu().numpy().squeeze()
    base_faces = model.faces
    base_weights = model.lbs_weights.detach().cpu().numpy()
    joints = output.joints.detach().cpu().numpy().squeeze()
    
    # Optional Upscale
    if subdivision_iterations > 0:
        print(f"Subdividing mesh {subdivision_iterations} time(s)")
        
        # Pack weights and UVs into a dictionary
        attrs = {'weights': base_weights}
        if uv_coords is not None:
            attrs['uvs'] = uv_coords
            
        up_verts, up_faces, new_attrs = ms_utils.subdivide_by_attributes(
            base_vertices, base_faces, attrs, iterations=subdivision_iterations
        )
        up_weights = new_attrs['weights']
        up_uvs = new_attrs.get('uvs', None)
    else:
        up_verts, up_faces, up_weights = base_vertices, base_faces, base_weights
        up_uvs = uv_coords
    
    # Enforce Symmetry
    print("Enforcing bilateral symmetry")
    v_mirror = ms_utils.find_vertex_symmetry(up_verts)
    j_mirror = ms_utils.get_joint_symmetry_map(names)
    sym_weights = ms_utils.make_symmetric_weights(up_weights, v_mirror, j_mirror)
    
    # Segment using Symmetric Weights
    segmented_parts = ms_utils.segment_by_provided_weights(
        names=names, 
        segment_joints=segment_joints, 
        pointcloud=up_verts, 
        lbs_weights=sym_weights # Use the symmetrized version!
    )
    
    if plotting:
        print("Plotting segments")
        ms_plot.plot_segments(segmented_parts, up_verts)
    
    if save:
        if os.path.exists(stl_folder):
            shutil.rmtree(stl_folder)
        os.mkdir(stl_folder)
    
    # Generate Convex Hulls
    print("Generating convex hulls")
    if save:
        print(f"Saving .stl files to {stl_folder}")
        
    hulls_dict = {}
    for key in segmented_parts.keys():
        collision_mesh = trimesh.convex.convex_hull(segmented_parts[key])
        collision_mesh.density = density_kg_per_m3
        hulls_dict[key] = collision_mesh
        
        if save:
            collision_mesh.export(os.path.join(stl_folder, f"{key}.stl"))
    
    if plotting:
        print("Plotting convex hulls")
        ms_plot.plot_collision_hulls(hulls_dict)
    
    # MuJoCo File
    if save:
        print("Building MuJoCo Kinematic Tree and Skin")
        ms_utils.generate_full_body_mjcf(
            network=network,
            names=names,
            joints=joints,
            segment_joints=segment_joints,
            pointcloud=up_verts,
            faces=up_faces,
            lbs_weights=sym_weights,
            uv_coords=up_uvs,
            texture_file=texture_file,
            stl_folder=stl_folder,
            output_file=output_file)
        
        
def vrm2mjcf(file_path: str,
             output_file: str = "vroid_model.xml",
             body_idx: int = 1,
             skin_index: int = 0,
             density_kg_per_m3: float = 1000.0) -> mv_utils.VRM:

    vrm_model = mv_utils.VRM(file_path)
    
    print(f"\nExtracting Data for Mesh {body_idx} (Body)...")
    verts, faces, bone_indices, bone_weights = vrm_model.extract_mesh_skinning_data(body_idx)
    
    print("Segmenting body pointcloud by dominant joint...")
    segmented_parts = vrm_model.segment_by_dominant_joint(
        skin_index=skin_index,
        vertices=verts,
        bone_indices=bone_indices,
        bone_weights=bone_weights
        )
    print(f"Segmented into {len(segmented_parts)} distinct body parts.")
    
    print("Generating Trimesh Convex Hulls...")
    hulls_dict = vrm_model.generate_convex_hulls(segmented_parts,
                                                 density_kg_per_m3)
    print(f"Successfully generated {len(hulls_dict)} physics hulls.")
    
    print("\nGenerating final MuJoCo XML...")
    # Pass the raw vertices so we can still calculate the minimum Z floor height
    vrm_model.generate_mjcf(
        skin_index=skin_index,
        raw_vertices=verts,
        hulls_dict=hulls_dict,
        output_file=output_file
        )
    print(f"\nGenerated MuJoCo XML at: {output_file}")
    
    return vrm_model

def vrm_sim(output_file: str,
            vrm_model: mv_utils.VRM,
            body_idx: int = 1,
            skin_index: int = 0,
            runtime: int = 300):

    # Initialize model to check compilation before streaming
    model = mujoco.MjModel.from_xml_path(output_file)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    print(f"Physics Model Compiled Successfully! Tree contains {model.nbody} bodies and {model.njnt} joints.")
    
    print("\nInitializing Physics Streamer...")
    vrm_model.start_physics_stream(
        skin_index=skin_index,
        output_file=output_file,
        runtime=runtime
    )