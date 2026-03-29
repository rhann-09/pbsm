#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Utility functions for the pbsm.mujoco_smplx sub-package.
"""

# %% Imports

# Standard Imports
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple

# Third-Party Imports
import smplx
import torch
import networkx as nx
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

torch.set_default_device('cpu')

# %% Functions

def make_name_and_network() -> Tuple[List[str], nx.Graph]:
    """
    Creates a standardized list of SMPL-X joint names and a kinematic tree network.

    This function defines the hierarchical skeleton of the SMPL-X model and maps 
    it to a NetworkX graph to represent the physical connections between joints.

    Returns
    -------
    names : List[str]
        A comprehensive list of joint names in the SMPL-X format.
    g : nx.Graph
        A NetworkX graph representing the edges (bones) connecting the joints.
    """
    
    names = [
    "pelvis","left_hip","right_hip",
    "spine1","left_knee","right_knee","spine2","left_ankle","right_ankle","spine3","left_foot","right_foot","neck",
    "left_collar","right_collar","head","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist",
    "right_wrist","jaw","left_eye_smplhf","right_eye_smplhf","left_index1","left_index2","left_index3","left_middle1",
    "left_middle2","left_middle3","left_pinky1","left_pinky2","left_pinky3","left_ring1","left_ring2","left_ring3",
    "left_thumb1","left_thumb2","left_thumb3","right_index1","right_index2","right_index3","right_middle1",
    "right_middle2","right_middle3","right_pinky1","right_pinky2","right_pinky3","right_ring1","right_ring2",
    "right_ring3","right_thumb1","right_thumb2","right_thumb3","nose","right_eye","left_eye","right_ear",
    "left_ear","left_big_toe","left_small_toe","left_heel","right_big_toe","right_small_toe","right_heel",
    "left_thumb","left_index","left_middle","left_ring","left_pinky","right_thumb","right_index","right_middle",
    "right_ring","right_pinky","right_eye_brow1","right_eye_brow2","right_eye_brow3","right_eye_brow4",
    "right_eye_brow5","left_eye_brow5","left_eye_brow4","left_eye_brow3","left_eye_brow2","left_eye_brow1",
    "nose1","nose2","nose3","nose4","right_nose_2","right_nose_1","nose_middle","left_nose_1","left_nose_2",
    "right_eye1","right_eye2","right_eye3","right_eye4","right_eye5","right_eye6","left_eye4","left_eye3","left_eye2",
    "left_eye1","left_eye6","left_eye5","right_mouth_1","right_mouth_2","right_mouth_3","mouth_top","left_mouth_3",
    "left_mouth_2","left_mouth_1","left_mouth_5", "left_mouth_4", "mouth_bottom","right_mouth_4","right_mouth_5",
    "right_lip_1","right_lip_2","lip_top","left_lip_2","left_lip_1","left_lip_3","lip_bottom","right_lip_3",
    ]

    g = nx.Graph()
    network_names = {}
    for idx, joint in enumerate(names):
        network_names[str(idx)] = joint


    g.nodes(network_names)

    # left leg
    g.add_edge("pelvis", "left_hip")
    g.add_edge("left_hip", "left_knee")
    g.add_edge("left_knee", "left_ankle")
    g.add_edge("left_ankle", "left_foot")
    g.add_edge("left_foot", "left_big_toe")
    g.add_edge("left_foot", "left_small_toe")
    g.add_edge("left_ankle", "left_heel")

    # right leg
    g.add_edge("pelvis", "right_hip")
    g.add_edge("right_hip", "right_knee")
    g.add_edge("right_knee", "right_ankle")
    g.add_edge("right_ankle", "right_foot")
    g.add_edge("right_foot", "right_big_toe")
    g.add_edge("right_foot", "right_small_toe")
    g.add_edge("right_ankle", "right_heel")

    # spine
    g.add_edge("pelvis", "spine1")
    g.add_edge("spine1", "spine2")
    g.add_edge("spine2", "spine3")
    g.add_edge("spine3", "neck")
    g.add_edge("neck", "head")
    g.add_edge("head", "jaw")

    # left arm
    g.add_edge("spine3", "left_collar")
    g.add_edge("left_collar", "left_shoulder")
    g.add_edge("left_shoulder", "left_elbow")
    g.add_edge("left_elbow", "left_wrist")
    g.add_edge("left_wrist", "left_index1")
    g.add_edge("left_index1", "left_index2")
    g.add_edge("left_index2", "left_index3")
    g.add_edge("left_index3", "left_index")
    g.add_edge("left_wrist", "left_middle1")
    g.add_edge("left_middle1", "left_middle2")
    g.add_edge("left_middle2", "left_middle3")
    g.add_edge("left_middle3", "left_middle")
    g.add_edge("left_wrist", "left_pinky1")
    g.add_edge("left_pinky1", "left_pinky2")
    g.add_edge("left_pinky2", "left_pinky3")
    g.add_edge("left_pinky3", "left_pinky")
    g.add_edge("left_wrist", "left_ring1")
    g.add_edge("left_ring1", "left_ring2")
    g.add_edge("left_ring2", "left_ring3")
    g.add_edge("left_pinky3", "left_pinky")
    g.add_edge("left_wrist", "left_thumb1")
    g.add_edge("left_thumb1", "left_thumb2")
    g.add_edge("left_thumb2", "left_thumb3")
    g.add_edge("left_thumb3", "left_thumb")

    # right arm
    g.add_edge("spine3", "right_collar")
    g.add_edge("right_collar", "right_shoulder")
    g.add_edge("right_shoulder", "right_elbow")
    g.add_edge("right_elbow", "right_wrist")
    g.add_edge("right_wrist", "right_index1")
    g.add_edge("right_index1", "right_index2")
    g.add_edge("right_index2", "right_index3")
    g.add_edge("right_index3", "right_index")
    g.add_edge("right_wrist", "right_middle1")
    g.add_edge("right_middle1", "right_middle2")
    g.add_edge("right_middle2", "right_middle3")
    g.add_edge("right_middle3", "right_middle")
    g.add_edge("right_wrist", "right_pinky1")
    g.add_edge("right_pinky1", "right_pinky2")
    g.add_edge("right_pinky2", "right_pinky3")
    g.add_edge("right_pinky3", "right_pinky")
    g.add_edge("right_wrist", "right_ring1")
    g.add_edge("right_ring1", "right_ring2")
    g.add_edge("right_ring2", "right_ring3")
    g.add_edge("right_ring3", "right_ring")
    g.add_edge("right_wrist", "right_thumb1")
    g.add_edge("right_thumb1", "right_thumb2")
    g.add_edge("right_thumb2", "right_thumb3")
    g.add_edge("right_thumb3", "right_thumb")

    return names, g


def default_smplx_model(
    model_path: str,
    gender: str,
    ext: str,
    betas: torch.Tensor,
    use_pca: bool = False,
    flat_hand_mean: bool = True,
    ) -> smplx.SMPLX:
    """
    Initialises and returns a default SMPL-X parametric body model.

    Parameters
    ----------
    model_path : str
        Path to the SMPL-X model weights file.
    gender : str
        Gender of the model ('male', 'female', or 'neutral').
    ext : str
        File extension of the model weights ("pkl" or "npz").
    betas : str
        Body shape tensor of type tensor.float32 and shape [1, 10] .
    use_pca : bool, optional
        Whether to use PCA for hand articulation, by default False.
    flat_hand_mean : bool, optional
        Whether to initialize hands in a flat pose, by default True.

    Returns
    -------
    smplx.SMPLX
        The instantiated SMPL-X model.
    """
    
    model = smplx.SMPLX(
        model_path,
        gender=gender,
        ext=ext,
        betas=betas,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        )
    return model


def find_vertex_symmetry(vertices: np.ndarray) -> np.ndarray:
    """
    Finds the index mapping for symmetric vertices in the SMPL-X template.

    Creates a mirrored version of the point cloud across the X-axis and uses a KDTree 
    to map each vertex to its corresponding mirror index.

    Parameters
    ----------
    vertices : np.ndarray
        A (N, 3) array of vertex coordinates.

    Returns
    -------
    np.ndarray
        A 1D array where the value at index `i` is the index of its mirrored vertex.
    """
    from scipy.spatial import cKDTree
    
    # Create a mirrored version of the point cloud (flip X axis)
    mirrored_verts = vertices.copy()
    mirrored_verts[:, 0] *= -1
    
    # Use a KDTree to find the nearest neighbor for every mirrored point
    tree = cKDTree(vertices)
    _, mirror_map = tree.query(mirrored_verts, k=1)
    
    return mirror_map


def get_joint_symmetry_map(names: List[str]) -> Dict[int, int]:
    """
    Creates a bidirectional mapping between left and right joint indices.

    Central joints (e.g., spine, pelvis) are mapped to themselves.

    Parameters
    ----------
    names : List[str]
        The full list of SMPL-X joint names.

    Returns
    -------
    Dict[int, int]
        A dictionary mapping the integer index of a joint to the integer index of its symmetric counterpart.
    """
    joint_map = {}
    for i, name in enumerate(names):
        if name.startswith("left_"):
            right_name = name.replace("left_", "right_")
            if right_name in names:
                j = names.index(right_name)
                joint_map[i] = j
                joint_map[j] = i
        elif name.startswith("right_"):
            continue # Already handled by left_ check
        else:
            # Central joints (spine, pelvis, etc.) map to themselves
            joint_map[i] = i
    return joint_map


def symmetrize_weights(lbs_weights: np.ndarray, 
                       v_mirror_map: np.ndarray, 
                       j_mirror_map: Dict[int, int]) -> np.ndarray:
    """
    Averages LBS weights across the sagittal plane to enforce strict bilateral symmetry.

    Parameters
    ----------
    lbs_weights : np.ndarray
        A (N, J) array of original Linear Blend Skinning weights.
    v_mirror_map : np.ndarray
        An array mapping each vertex index to its symmetric counterpart.
    j_mirror_map : Dict[int, int]
        A dictionary mapping each joint index to its symmetric counterpart.

    Returns
    -------
    np.ndarray
        A (N, J) array of symmetrized LBS weights.
    """    
    
    sym_weights = np.zeros_like(lbs_weights)
    
    for i in range(len(lbs_weights)):
        mirror_v_idx = v_mirror_map[i]
        
        for j in range(lbs_weights.shape[1]):
            mirror_j_idx = j_mirror_map[j]
            
            # The weight of vertex i for joint j should match 
            # the weight of vertex mirror_i for joint mirror_j
            val_a = lbs_weights[i, j]
            val_b = lbs_weights[mirror_v_idx, mirror_j_idx]
            
            avg_weight = (val_a + val_b) / 2.0
            sym_weights[i, j] = avg_weight
            
    return sym_weights


def subdivide_with_weights(vertices: np.ndarray, 
                           faces: np.ndarray, 
                           lbs_weights: np.ndarray, 
                           iterations: int = 1) -> Tuple[np.ndarray]:
    """
    Subdivides the mesh while interpolating Linear Blend Skinning weights.

    Parameters
    ----------
    vertices : np.ndarray
        A (N, 3) array of initial vertex coordinates.
    faces : np.ndarray
        A (F, 3) array of initial face indices.
    lbs_weights : np.ndarray
        A (N, J) array of initial vertex weights.
    iterations : int, optional
        The number of subdivision iterations to perform, by default 1.

    Returns
    -------
    current_v : np.ndarray
        The upsampled (M, 3) vertex array.
    current_f : np.ndarray
        The upsampled (K, 3) face array.
    current_w : np.ndarray
        The interpolated (M, J) weights array corresponding to the new vertices.
    """
    
    current_v = vertices
    current_f = faces
    current_w = lbs_weights
    
    for _ in range(iterations):
        # trimesh automatically handles the barycentric/midpoint interpolation 
        # for any arrays passed into vertex_attributes!
        current_v, current_f, attributes = trimesh.remesh.subdivide(
            vertices=current_v,
            faces=current_f,
            vertex_attributes={'weights': current_w}
        )
        current_w = attributes['weights']
        
    return current_v, current_f, current_w


def segment_by_provided_weights(names: List[str],
                                segment_joints: List[str],
                                pointcloud: np.ndarray,
                                lbs_weights: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Segments the pointcloud based on a provided array of LBS weights.

    Finds the dominant joint for each vertex based on the provided weight matrix 
    and groups the vertices accordingly.

    Parameters
    ----------
    names : List[str]
        The full list of SMPL-X joint names.
    segment_joints : List[str]
        The subset of joint names to use for segmentation.
    pointcloud : np.ndarray
        A (N, 3) array of vertex coordinates.
    lbs_weights : np.ndarray
        A (N, J) array of calculated or modified LBS weights.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary mapping joint names to their corresponding (V, 3) vertex arrays.
    """
    # For every vertex, find the index of the joint that influences it the most
    dominant_joint_indices = np.argmax(lbs_weights, axis=1)

    segmented_parts = {}

    for joint_name in segment_joints:
        if joint_name in names:
            joint_idx = names.index(joint_name)
            vertex_mask = (dominant_joint_indices == joint_idx)
            part_vertices = pointcloud[vertex_mask]
            segmented_parts[joint_name] = part_vertices

    return segmented_parts


def generate_full_body_mjcf(network: nx.Graph,
                            names: List[str],
                            joints: np.ndarray,
                            segment_joints: List[str],
                            pointcloud: np.ndarray,
                            faces: np.ndarray,          
                            lbs_weights: np.ndarray,    
                            stl_folder: str = "STL",
                            output_file: str = "smplx_full_body.xml") -> None:
    """
    Generates a complete MuJoCo XML (MJCF) file for the segmented SMPL-X physics body.

    Builds the kinematic tree, handles skin attachment, assigns collision meshes 
    from STL files, and manages weight normalizations for the physics engine.

    Parameters
    ----------
    network : nx.Graph
        A NetworkX graph defining the parent-child relationships between bones.
    names : List[str]
        The full list of SMPL-X joint names.
    joints : np.ndarray
        A (J, 3) array of global joint positions.
    segment_joints : List[str]
        The list of joint names acting as active physics bodies.
    pointcloud : np.ndarray
        A (N, 3) array of the overall body vertices for the skin mesh.
    faces : np.ndarray
        A (F, 3) array of face indices for the skin mesh.
    lbs_weights : np.ndarray
        A (N, J) array of vertex skinning weights.
    stl_folder : str, optional
        The directory containing the generated STL collision meshes, by default "STL".
    output_file : str, optional
        The desired filename for the output MJCF XML, by default "smplx_full_body.xml".

    Returns
    -------
    None
    """

    mujoco = ET.Element("mujoco", model="smplx_physics_body")
    
    # Compiler and Defaults
    ET.SubElement(mujoco, "compiler", angle="radian", meshdir=stl_folder, autolimits="true")
    
    default = ET.SubElement(mujoco, "default")
    ET.SubElement(default, "geom", type="mesh", density="1000", rgba="0.6 0.8 0.9 0.0", contype="1", conaffinity="1")
    ET.SubElement(default, "joint", type="hinge", range="-1.5708 1.5708", damping="0.1", stiffness="0.1")
    
    # Asset Registration
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "material", name="skin_mat", specular="0.2", shininess="0.1", rgba="0.8 0.6 0.5 1")
    for joint in segment_joints:
        ET.SubElement(asset, "mesh", name=joint, file=f"{joint}.stl")
        
    # Worldbody Setup
    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 5", dir="0 0 -1", directional="true")
    ET.SubElement(worldbody, "geom", type="plane", size="5 5 0.1", rgba="0.3 0.3 0.3 1", name="floor") 
    
    def f2s(arr): return f"{arr[0]:.6f} {arr[1]:.6f} {arr[2]:.6f}"
    
    # Recursive Tree Builder
    def build_tree(parent_elem: ET.Element, current_joint: str, parent_joint: str, parent_global_pos: np.ndarray):
        idx = names.index(current_joint)
        current_global_pos = joints[idx]
        
        rel_pos = current_global_pos - parent_global_pos
        body_elem = ET.SubElement(parent_elem, "body", name=current_joint, pos=f2s(rel_pos))
        
        ET.SubElement(body_elem, "joint", name=f"{current_joint}_x", axis="1 0 0")
        ET.SubElement(body_elem, "joint", name=f"{current_joint}_y", axis="0 1 0")
        ET.SubElement(body_elem, "joint", name=f"{current_joint}_z", axis="0 0 1")
        
        geom_pos = -current_global_pos
        ET.SubElement(body_elem, "geom", mesh=current_joint, pos=f2s(geom_pos))
        
        for neighbor in network.neighbors(current_joint):
            if neighbor != parent_joint and neighbor in segment_joints:
                build_tree(body_elem, neighbor, current_joint, current_global_pos)
                
    # Initialize the Root Body Upright & Calculate Skin Transform
    root_name = "pelvis"
    root_idx = names.index(root_name)
    root_global_pos = joints[root_idx]
    
    r = R.from_euler('xyz', (90, 0, 0), degrees=True)
    verts_local = pointcloud - root_global_pos
    verts_rotated = r.apply(verts_local)
    min_z_local = np.min(verts_rotated[:, 2])
    spawn_pos = np.array([0.0, 0.0, -min_z_local])
    
    skin_verts_qpos0 = verts_rotated + spawn_pos
    
    root_body = ET.SubElement(worldbody, "body", name=root_name, pos=f2s(spawn_pos), euler="1.570796 0 0")
    ET.SubElement(root_body, "freejoint", name="root_freejoint")
    ET.SubElement(root_body, "geom", mesh=root_name, pos=f2s(-root_global_pos))
    
    for neighbor in network.neighbors(root_name):
        if neighbor in segment_joints:
            build_tree(root_body, neighbor, root_name, root_global_pos)
            
    # Generate the <skin> Element
    skin_elem = ET.SubElement(asset, "skin", name="smplx_skin", material="skin_mat")
    skin_elem.set("vertex", " ".join([f"{v:.5f}" for v in skin_verts_qpos0.flatten()]))
    skin_elem.set("face", " ".join([str(f) for f in faces.flatten()]))
    
    # Weight clean-up pass
    from scipy.spatial import cKDTree
    
    seg_idx = [names.index(j) for j in segment_joints]
    active_weights = lbs_weights[:, seg_idx].copy()
    
    # Filter out negligible weights
    active_weights[active_weights < 0.01] = 0.0
    
    row_sums = active_weights.sum(axis=1)
    orphan_indices = np.where(row_sums == 0)[0]
    
    # If any vertices have 0 weight, map them to the closest physical bone
    if len(orphan_indices) > 0:
        tree = cKDTree(joints[seg_idx])
        _, closest_j_idx = tree.query(pointcloud[orphan_indices])
        
        active_weights[orphan_indices, closest_j_idx] = 1.0
        row_sums[orphan_indices] = 1.0
        
    # Normalize so every vertex has exactly 1.0 total weight
    active_weights = active_weights / row_sums[:, np.newaxis]

    # Bind bones
    for i, joint_name in enumerate(segment_joints):
        joint_weights = active_weights[:, i]
        influenced_verts = np.where(joint_weights > 0)[0]
        
        if len(influenced_verts) == 0:
            continue
            
        joint_idx = names.index(joint_name)
        raw_jpos = joints[joint_idx]
        
        # Transform the bindpos exactly like the skin vertices
        jpos_local = raw_jpos - root_global_pos
        jpos_rotated = r.apply(jpos_local)
        jpos_qpos0 = jpos_rotated + spawn_pos
        
        bone_elem = ET.SubElement(skin_elem, "bone", body=joint_name)
        bone_elem.set("bindpos", f2s(jpos_qpos0))
        bone_elem.set("bindquat", "0.7071068 0.7071068 0.0 0.0")
        bone_elem.set("vertid", " ".join(map(str, influenced_verts)))
        bone_elem.set("vertweight", " ".join([f"{w:.4f}" for w in joint_weights[influenced_verts]]))

    # Format and Write XML
    xml_str = ET.tostring(mujoco, encoding='utf-8')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
    pretty_xml = os.linesep.join([s for s in pretty_xml.splitlines() if s.strip()])
    
    with open(output_file, "w") as f:
        f.write(pretty_xml)
        
    print(f"Generated full-body MJCF at: {output_file}")
