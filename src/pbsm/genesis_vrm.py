#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

VRM modelling within Genesis and Three.js.
"""

# %% Imports

# Standard Library Imports
import os
import shutil
import threading
import http.server
import socketserver
import webbrowser
import websockets
import time
import json
import asyncio
import socket
import functools
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, List, Any, Optional
from xml.dom import minidom

# Third-Party Imports
import networkx as nx
import numpy as np
import pygltflib
import trimesh
import genesis as gs
import torch
from jinja2 import Template
from scipy.spatial.transform import Rotation as R


# %% Classes

class VRM(object):
    """
    A class for parsing .vrm files into MJCF files.

    Parameters
    ----------
    file_path : str
        File path to the input .vrm file.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialise the VRM class.

        Parses the .vrm file as a glTF file using pygltflib and extracts
        available skinning/skeleton data.

        Parameters
        ----------
        file_path : str
            File path to the input .vrm file.
        """
        self.file_path = file_path
        self.gltf = self._load_and_inspect(file_path)
        self.skins_data = []

        if self.gltf.skins:
            for skin_index in range(len(self.gltf.skins)):
                names, network, joints = self.extract_vrm_skeleton(skin_index)
                self.skins_data.append({
                    "names": names,
                    "network": network,
                    "global_joints": joints,
                })


    def extract_vrm_skeleton(
            self,
            skin_index: int = 0,
            ) -> Tuple[List[str], nx.DiGraph, np.ndarray]:
        """
        Extracts the kinematic tree and global joint positions.

        The kinematic tree for all parts of the .vrm file is the same, so it
        will default to reading from the first skin part.

        Parameters
        ----------
        skin_index : int, optional
            Index of the part to read the kinematic tree from in the .vrm file.
            Default is 0.

        Returns
        -------
        names : List[str]
            List of node names of the kinematic tree.
        network : nx.DiGraph
            Network graph of the kinematic tree.
        global_joints : np.ndarray
            Global Cartesian positions of each joint as a float array of shape (N, 3).
        """
        skin = self.gltf.skins[skin_index]
        joint_indices = skin.joints

        names = []
        network = nx.DiGraph()
        node_to_jidx = {}

        for idx, node_idx in enumerate(joint_indices):
            node = self.gltf.nodes[node_idx]
            name = node.name if node.name else f"bone_{node_idx}"
            names.append(name)
            node_to_jidx[node_idx] = idx
            network.add_node(name)

        for node_idx in joint_indices:
            node = self.gltf.nodes[node_idx]
            if not node.children:
                continue

            parent_name = names[node_to_jidx[node_idx]]
            for child_idx in node.children:
                if child_idx in joint_indices:
                    child_name = names[node_to_jidx[child_idx]]
                    network.add_edge(parent_name, child_name)

        roots = [n for n, d in network.in_degree() if d == 0]
        global_joints = np.zeros((len(names), 3))

        def compute_global_pos(current_name: str, parent_global_pos: np.ndarray):
            jidx = names.index(current_name)
            node_idx = joint_indices[jidx]
            node = self.gltf.nodes[node_idx]

            local_trans = np.array(node.translation) if node.translation else np.array([0.0, 0.0, 0.0])
            current_global_pos = parent_global_pos + local_trans
            global_joints[jidx] = current_global_pos

            for child_name in network.successors(current_name):
                compute_global_pos(child_name, current_global_pos)

        for root_name in roots:
            compute_global_pos(root_name, np.array([0.0, 0.0, 0.0]))

        return names, network, global_joints


    def extract_mesh_skinning_data(
            self,
            mesh_index: int,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts and merges vertices, faces, joints, and LBS weights for a mesh.

        Parameters
        ----------
        mesh_index : int
            Index of the mesh to be extracted from the glTF file.

        Returns
        -------
        vertices_out : np.ndarray
            Vertices array of shape (V, 3).
        faces_out : np.ndarray
            Faces (indices) array of shape (F, 3).
        joints_out : np.ndarray
            Joint indices array of shape (V, 4) mapping vertices to bones.
        weights_out : np.ndarray
            Linear Blend Skinning (LBS) influence weights array of shape (V, 4).
        """
        mesh = self.gltf.meshes[mesh_index]

        all_vertices = []
        all_faces = []
        all_joint_indices = []
        all_weights = []
        vertex_offset = 0

        for primitive in mesh.primitives:
            if primitive.attributes.POSITION is not None:
                verts = self._get_accessor_data(primitive.attributes.POSITION)
                all_vertices.append(verts)

            if primitive.indices is not None:
                faces = self._get_accessor_data(primitive.indices).reshape(-1, 3)
                all_faces.append(faces + vertex_offset)

            if primitive.attributes.JOINTS_0 is not None:
                all_joint_indices.append(self._get_accessor_data(primitive.attributes.JOINTS_0))

            if primitive.attributes.WEIGHTS_0 is not None:
                all_weights.append(self._get_accessor_data(primitive.attributes.WEIGHTS_0))

            vertex_offset += len(verts)

        vertices_out = np.vstack(all_vertices) if all_vertices else np.array([])
        faces_out = np.vstack(all_faces) if all_faces else np.array([])
        joints_out = np.vstack(all_joint_indices) if all_joint_indices else np.array([])
        weights_out = np.vstack(all_weights) if all_weights else np.array([])

        return vertices_out, faces_out, joints_out, weights_out


    def segment_by_dominant_joint(
            self,
            skin_index: int,
            vertices: np.ndarray,
            bone_indices: np.ndarray,
            bone_weights: np.ndarray,
            ) -> Dict[str, np.ndarray]:
        """
        Segment the mesh by the node (bone) that has the highest influence.

        Groups mesh vertices together by the node/bone that has the largest
        linear blended skinning weight, offsetting them to the joint's local space.

        Parameters
        ----------
        skin_index : int
            Index of the skin reference used to extract joints.
        vertices : np.ndarray
            Array of mesh vertices.
        bone_indices : np.ndarray
            Array of node indices affecting the vertices.
        bone_weights : np.ndarray
            Array of node LBS weights affecting the vertices.

        Returns
        -------
        segmented_parts : Dict[str, np.ndarray]
            Dictionary mapping the node string name to a NumPy array of its segmented,
            locally-offset vertices.
        """
        names = self.skins_data[skin_index]['names']
        joints = self.skins_data[skin_index]['global_joints']

        max_weight_cols = np.argmax(bone_weights, axis=1)
        dominant_bone_indices = bone_indices[np.arange(len(bone_indices)), max_weight_cols]

        segmented_parts = {}
        unique_bones = np.unique(dominant_bone_indices)

        for j_idx in unique_bones:
            joint_name = names[j_idx]
            mask = (dominant_bone_indices == j_idx)

            localized_verts = vertices[mask] - joints[j_idx]
            segmented_parts[joint_name] = localized_verts

        return segmented_parts


    def generate_convex_hulls(
            self,
            segmented_parts: Dict[str, np.ndarray],
            density: float = 1000.0,
            ) -> Dict[str, trimesh.Trimesh]:
        """
        Generate convex hulls for each segmented part.

        Creates 3D convex hulls for each segmented part of the body mesh, applying
        a density to calculate the physics inertia and mass of the body part.

        Parameters
        ----------
        segmented_parts : Dict[str, np.ndarray]
            Dictionary of segmented part names mapped to vertex pointclouds.
        density : float, optional
            Density of the generated hulls in kg/m^3. Default is 1000.0.

        Returns
        -------
        hulls_dict : Dict[str, trimesh.Trimesh]
            Dictionary mapping segmented part names to their trimesh convex hull objects.
        """
        hulls_dict = {}
        for joint_name, points in segmented_parts.items():
            try:
                hull = trimesh.convex.convex_hull(points)
                hull.density = density
                hulls_dict[joint_name] = hull
            except Exception as e:
                print(f"Could not generate hull for {joint_name}: {e}")

        return hulls_dict


    def generate_mjcf(
            self,
            skin_index: int,
            raw_vertices: np.ndarray,
            hulls_dict: Dict[str, trimesh.Trimesh],
            stl_folder: str = "STL",
            output_file: str = "vrm_physics_body.xml",
            density: int = 1000,
            damping: float = 2.0,
            stiffness: float = 0.01,
            friction: Tuple[float, float, float] = (1., 0.005, 0.0001),
            spawn: Tuple[float, float, float] = (0., 0., 0.),
            ) -> None:
        """
        Generates the pure physics-focused MuJoCo XML (MJCF) kinematic tree and exports hulls.

        Parameters
        ----------
        skin_index : int
            Index of the skin reference to construct the kinematic tree.
        raw_vertices : np.ndarray
            Array of raw vertex positions, used to calculate the physical floor offset.
        hulls_dict : Dict[str, trimesh.Trimesh]
            Dictionary mapping joint names to `trimesh.Trimesh` convex hulls.
        stl_folder : str, optional
            Directory to export the generated STL collision files. Default is "STL".
        output_file : str, optional
            Filename for the generated MJCF XML file. Default is "vrm_physics_body.xml".
        density : int, optional
            Density of the geometries. Default is 1000.
        damping : float, optional
            Joint damping parameter. Default is 2.0.
        stiffness : float, optional
            Joint stiffness parameter. Default is 0.01.
        friction : Tuple[float, float, float], optional
            Friction tuple. Default is (1., 0.005, 0.0001).
        spawn : Tuple[float, float, float], optional
            Spawn offset coordinate. Default is (0., 0., 0.).
        """
        if os.path.exists(stl_folder):
            shutil.rmtree(stl_folder)
        os.mkdir(stl_folder)

        for name, hull in hulls_dict.items():
            hull.export(os.path.join(stl_folder, f"{name}.stl"))

        Mujoco = ET.Element("mujoco", model="vrm_physics_body")
        ET.SubElement(Mujoco, "compiler", angle="radian", meshdir=stl_folder, autolimits="true")

        default = ET.SubElement(Mujoco, "default")
        ET.SubElement(default, "geom", density=f"{density}", contype="1", conaffinity="0", friction=f"{friction[0]} {friction[1]} {friction[2]}")
        ET.SubElement(default, "joint", damping=f"{damping}", stiffness=f"{stiffness}")

        asset = ET.SubElement(Mujoco, "asset")
        for name in hulls_dict.keys():
            ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl")

        worldbody = ET.SubElement(Mujoco, "worldbody")

        def f2s(arr):
            return f"{arr[0]:.6f} {arr[1]:.6f} {arr[2]:.6f}"

        names = self.skins_data[skin_index]['names']
        network = self.skins_data[skin_index]['network']
        joints = self.skins_data[skin_index]['global_joints']

        roots = [n for n, d in network.in_degree() if d == 0]
        root_name = roots[0]

        while len(list(network.successors(root_name))) == 1:
            root_name = list(network.successors(root_name))[0]

        root_global_pos = joints[names.index(root_name)]

        verts_local = raw_vertices - root_global_pos
        min_y_local = np.min(verts_local[:, 1])
        spawn_pos = np.array([spawn[0], -min_y_local + spawn[1], spawn[2]])

        def build_tree(parent_elem: ET.Element, current_joint: str, parent_global_pos: np.ndarray):
            idx = names.index(current_joint)
            current_global_pos = joints[idx]
            rel_pos = current_global_pos - parent_global_pos
            body_elem = ET.SubElement(parent_elem, "body", name=current_joint, pos=f2s(rel_pos))

            ET.SubElement(body_elem, "joint", name=current_joint, type="ball")

            if current_joint in hulls_dict:
                ET.SubElement(body_elem, "geom", type="mesh", mesh=current_joint)

            ET.SubElement(body_elem, "geom", type="sphere", size="0.05", mass="0.5",
                          contype="0", conaffinity="0", rgba="0 0 0 0")

            for child in network.successors(current_joint):
                build_tree(body_elem, child, current_global_pos)

        root_body = ET.SubElement(worldbody, "body", name=root_name, pos=f2s(spawn_pos))
        ET.SubElement(root_body, "freejoint", name="root_freejoint")

        if root_name in hulls_dict:
            ET.SubElement(root_body, "geom", type="mesh", mesh=root_name)

        ET.SubElement(root_body, "geom", type="sphere", size="0.05", mass="0.5",
                      contype="0", conaffinity="0", rgba="0 0 0 0")

        for child in network.successors(root_name):
            build_tree(root_body, child, root_global_pos)

        xml_str = ET.tostring(Mujoco, encoding='utf-8')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
        pretty_xml = os.linesep.join([s for s in pretty_xml.splitlines() if s.strip()])

        with open(output_file, "w") as f:
            f.write(pretty_xml)


    def vrm2mjcf(
            self,
            output_file: str = "vroid_model.xml",
            body_idx: int = 1,
            skin_index: int = 0,
            density_kg_per_m3: float = 1000.0
            ) -> None:
        """
        Processes a VRM model to generate collision hulls and a MuJoCo MJCF XML file.

        Parameters
        ----------
        output_file : str, optional
            Filename for the generated MuJoCo XML file. Default is "vroid_model.xml".
        body_idx : int, optional
            Index of the primary mesh to extract skinning data from. Default is 1.
        skin_index : int, optional
            Index of the skin reference to use for the kinematic tree. Default is 0.
        density_kg_per_m3 : float, optional
            Density of the generated convex hulls in kg/m^3. Default is 1000.0.
        """
        print(f"Extracting Data for Mesh {body_idx} (Body)")
        verts, faces, bone_indices, bone_weights = self.extract_mesh_skinning_data(body_idx)

        print("Segmenting body pointcloud by dominant joint")
        segmented_parts = self.segment_by_dominant_joint(
            skin_index=skin_index,
            vertices=verts,
            bone_indices=bone_indices,
            bone_weights=bone_weights
            )
        print(f"Segmented into {len(segmented_parts)} distinct body parts")

        print("Generating Trimesh Convex Hulls")
        hulls_dict = self.generate_convex_hulls(segmented_parts, density_kg_per_m3)
        print(f"Successfully generated {len(hulls_dict)} physics hulls")

        print("Generating final MuJoCo XML")
        self.generate_mjcf(
            skin_index=skin_index,
            raw_vertices=verts,
            hulls_dict=hulls_dict,
            output_file=output_file
            )
        print(f"Generated MuJoCo XML at: {output_file}")


    def _load_and_inspect(self, file_path: str) -> pygltflib.GLTF2:
        """
        Loads the .vrm binary file using pygltflib.

        Parameters
        ----------
        file_path : str
            Path to the target VRM file.

        Returns
        -------
        pygltflib.GLTF2
            The parsed glTF object.
        """
        gltf = pygltflib.GLTF2().load_binary(file_path)
        print(f"Successfully loaded: {file_path}")
        return gltf


    def _get_accessor_data(self, accessor_idx: int) -> Optional[np.ndarray]:
        """
        Reads raw binary glTF buffer data and converts it into a Numpy array.

        Parameters
        ----------
        accessor_idx : int
            Index of the accessor within the glTF file.

        Returns
        -------
        np.ndarray or None
            The formatted numeric array corresponding to the accessor data,
            or None if the index is invalid.
        """
        if accessor_idx is None:
            return None

        accessor = self.gltf.accessors[accessor_idx]
        buffer_view = self.gltf.bufferViews[accessor.bufferView]
        binary_data = self.gltf.binary_blob()

        component_type_map = {
            5120: np.int8, 5121: np.uint8, 5122: np.int16,
            5123: np.uint16, 5125: np.uint32, 5126: np.float32
        }
        data_type_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

        dtype = component_type_map[accessor.componentType]
        num_components = data_type_map[accessor.type]

        start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
        item_size = np.dtype(dtype).itemsize
        stride = buffer_view.byteStride

        if stride is None or stride == 0 or stride == item_size * num_components:
            count = accessor.count * num_components
            end = start + count * item_size
            raw_bytes = binary_data[start:end]
            array = np.frombuffer(raw_bytes, dtype=dtype).reshape(accessor.count, num_components).copy()
        else:
            end = start + (accessor.count - 1) * stride + num_components * item_size
            raw_bytes = binary_data[start:end]
            array = np.lib.stride_tricks.as_strided(
                np.frombuffer(raw_bytes, dtype=dtype),
                shape=(accessor.count, num_components),
                strides=(stride, item_size)
            ).copy()

        if num_components == 1:
            array = array.flatten()

        if getattr(accessor, 'normalized', False):
            if dtype == np.int8:
                array = np.maximum(array / 127.0, -1.0)
            elif dtype == np.uint8:
                array = array / 255.0
            elif dtype == np.int16:
                array = np.maximum(array / 32767.0, -1.0)
            elif dtype == np.uint16:
                array = array / 65535.0
            array = array.astype(np.float32)

        return array


class Entity:
    """
    Data holder for creating scene entities in a Genesis Physics Env.

    Parameters
    ----------
    morphs_type : str
        The geometry type of the morphology. Expected values are "MJCF",
        "MESH", "BOX", "CYLINDER", or "SPHERE" (case-insensitive).
    file : str | None, optional
        The file path to the asset. Required if `morphs_type` is "MJCF" or "MESH". Default is None.
    vrm_file : str | None, optional
        The file path to the original .vrm model (used by the Bridge). Default is None.
    vrm_obj : VRM | None, optional
        The instantiated VRM class object parsed from `vrm_file` (used by the Bridge). Default is None.
    pos : Tuple[float, float, float], optional
        The (x, y, z) position coordinates of the entity. Default is (0.0, 0.0, 0.0).
    euler : Tuple[int, int, int], optional
        The (roll, pitch, yaw) Euler angles for rotation. Default is (0, 0, 0).
    scale : float, optional
        The scaling factor applied to the entity. Default is 1.0.
    lower : Tuple[float, float, float], optional
        The lower bounds for the entity's bounding box. Default is (0.0, 0.0, 0.0).
    upper : Tuple[float, float, float], optional
        The upper bounds for the entity's bounding box. Default is (1.0, 1.0, 1.0).
    visualization : bool, optional
        Whether the entity should be rendered visually. Default is True.
    collision : bool, optional
        Whether the entity should participate in physics collisions. Default is True.
    radius : float, optional
        The radius dimension, applicable for "CYLINDER" or "SPHERE" types. Default is 0.5.
    height : float, optional
        The height dimension, applicable for "CYLINDER" types. Default is 1.0.

    Attributes
    ----------
    types : List[str]
        The list of supported morphology types.
    """
    def __init__(
        self,
        morphs_type: str,
        file: str | None = None,
        vrm_file: str | None = None,
        vrm_obj: Any | None = None,
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        euler: Tuple[int, int, int] = (0, 0, 0),
        scale: float = 1.0,
        lower: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        upper: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        visualization: bool = True,
        collision: bool = True,
        radius: float = 0.5,
        height: float = 1.0,
        ) -> None:
        """
        Initialises the Entity configuration object.
        """
        self.types = ["MJCF", "MESH", "BOX", "CYLINDER", "SPHERE"]
        file_req = ["MJCF", "MESH"]

        if morphs_type.upper() in file_req and file is None:
            raise ValueError(f"`file` must be a path to a file, not: {file}")

        if morphs_type.upper() in self.types:
            self.morphs_type = morphs_type.upper()
        else:
            raise ValueError(f"`morphs_type`: {morphs_type} not in {self.types}")

        self.file = file
        self.vrm_file = vrm_file
        self.vrm_obj = vrm_obj
        self.pos = pos
        self.euler = euler
        self.scale = scale
        self.lower = lower
        self.upper = upper
        self.visualization = visualization
        self.collision = collision
        self.radius = radius
        self.height = height


class GenesisSim:
    """
    Manager class for handling a Genesis physics simulation environment.

    This class encapsulates the configuration, initialization, entity management,
    and execution loop of a Genesis scene.

    Parameters
    ----------
    backend : Any, optional
        The compute backend to use for the simulation (e.g., gs.cpu, gs.gpu).
        Default is gs.cpu.
    precision : str, optional
        The floating-point precision for the simulation ('32' or '64'). Default is '32'.
    seed : int | None, optional
        Random seed for reproducibility. Default is None.
    debug : bool, optional
        Whether to enable debug mode in Genesis. Default is False.
    performance_mode : bool, optional
        Whether to optimize the engine for performance over flexibility. Default is False.
    logging_level : str | None, optional
        The verbosity level for Genesis logging (e.g., 'debug', 'warning'). Default is None.
    theme : str, optional
        The visual theme for the viewer (e.g., 'dark', 'light'). Default is "dark".
    logger_verbose_time : bool, optional
        Whether to include verbose timestamps in the logger. Default is False.
    dt : float, optional
        The time step size for the physics simulation. Default is 0.01.
    gravity : Tuple[float, float, float], optional
        The (x, y, z) gravity vector applied to the scene. Default is (0.0, 0.0, -10.0).
    show_world_frame : bool, optional
        Whether to render the global origin coordinate frame. Default is True.
    world_frame_size : float, optional
        The size of the rendered world coordinate frame. Default is 1.0.
    show_link_frame : bool, optional
        Whether to render the local coordinate frames of kinematic links. Default is False.
    show_cameras : bool, optional
        Whether to render camera frustums in the scene. Default is False.
    plane_reflection : bool, optional
        Whether the ground plane should be reflective. Default is True.
    ambient_light : Tuple[float, float, float], optional
        The RGB values for the ambient scene lighting. Default is (0.1, 0.1, 0.1).
    camera_pos : Tuple[float, float, float], optional
        The (x, y, z) initial position of the viewer camera. Default is (3.5, 0.0, 2.5).
    camera_lookat : Tuple[float, float, float], optional
        The (x, y, z) target focal point for the viewer camera. Default is (0.0, 0.0, 0.5).
    camera_fov : int, optional
        The field of view angle for the viewer camera in degrees. Default is 60.
    max_FPS : int, optional
        The maximum frames per second for the GUI viewer. Default is 60.
    res : Tuple[int, int], optional
        The (width, height) resolution of the viewer window. Default is (1280, 960).
    show_viewer : bool, optional
        Whether to launch the interactive GUI viewer. Default is True.
    """
    def __init__(
            self,
            backend: Any = gs.cpu,
            precision: str = '32',
            seed: int | None = None,
            debug: bool = False,
            performance_mode: bool = False,
            logging_level: str | None  = None,
            theme: str = "dark",
            logger_verbose_time: bool = False,
            dt: float = 0.01,
            gravity: Tuple[float, float, float] = (0., 0., -10.),
            show_world_frame: bool = True,
            world_frame_size: float = 1.,
            show_link_frame: bool = False,
            show_cameras: bool = False,
            plane_reflection: bool = True,
            ambient_light: Tuple[float, float, float] = (0.1, 0.1, 0.1),
            camera_pos: Tuple[float, float, float] = (3.5, 0.0, 2.5),
            camera_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.5),
            camera_fov: int = 60,
            max_FPS: int = 60,
            res: Tuple[int, int] = (1280, 960),
            show_viewer: bool = True,
            ) -> None:
        """
        Initialises the GenesisSim parameter configuration.
        """
        self.scene = None
        self.entities = {}

        self.backend = backend
        self.precision = precision
        self.seed = seed
        self.debug = debug
        self.performance_mode = performance_mode
        self.logging_level = logging_level
        self.logger_verbose_time = logger_verbose_time
        self.theme = theme

        self.dt = dt
        self.gravity = gravity
        self.show_world_frame = show_world_frame
        self.world_frame_size = world_frame_size
        self.show_link_frame = show_link_frame
        self.show_cameras = show_cameras
        self.plane_reflection = plane_reflection
        self.ambient_light = ambient_light
        self.camera_pos = camera_pos
        self.camera_lookat = camera_lookat
        self.camera_fov = camera_fov
        self.max_FPS = max_FPS
        self.res = res
        self.show_viewer = show_viewer


    def initialise(self) -> None:
        """
        Initialises the global Genesis framework using the configured instance parameters.
        """
        gs.init(
            backend = self.backend,
            precision = self.precision,
            seed = self.seed,
            debug = self.debug,
            performance_mode = self.performance_mode,
            logging_level = self.logging_level,
            theme = self.theme,
            logger_verbose_time = self.logger_verbose_time,
        )


    def create_scene(self) -> None:
        """
        Instantiates the Genesis scene object based on the initialized configuration.
        """
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(
                dt = self.dt,
                gravity = self.gravity,
                ),
            vis_options = gs.options.VisOptions(
                show_world_frame = self.show_world_frame,
                world_frame_size = self.world_frame_size,
                show_link_frame = self.show_link_frame,
                show_cameras = self.show_cameras,
                plane_reflection = self.plane_reflection,
                ambient_light = self.ambient_light,
                ),
            viewer_options = gs.options.ViewerOptions(
                res = self.res,
                camera_pos = self.camera_pos,
                camera_lookat = self.camera_lookat,
                camera_fov = self.camera_fov,
                max_FPS = self.max_FPS,
                ),
            show_viewer = self.show_viewer,
            )


    def add_entity(self, name: str, entity: Entity) -> None:
        """
        Adds a single morphological entity to the current Genesis scene.

        Parameters
        ----------
        name : str
            The internal reference name to store the entity under in `self.entities`.
        entity : Entity
            The configuration object detailing the entity's geometry type, file path,
            pose, and physical properties.
        """
        if entity.morphs_type == "MJCF":
            morphs = self.scene.add_entity(
                gs.morphs.MJCF(
                    file = entity.file,
                    pos = entity.pos,
                    euler = entity.euler,
                    scale = entity.scale,
                    )
                )
        elif entity.morphs_type == "URDF":
            morphs = self.scene.add_entity(
                gs.morphs.URDF(
                    file = entity.file,
                    pos = entity.pos,
                    euler = entity.euler,
                    scale = entity.scale,
                    )
                )
        elif entity.morphs_type == "MESH":
            morphs = self.scene.add_entity(
                gs.morphs.Mesh(
                    file = entity.file,
                    pos = entity.pos,
                    euler = entity.euler,
                    scale = entity.scale,
                    visualization = entity.visualization,
                    collision = entity.collision,
                    )
                )
        elif entity.morphs_type == "BOX":
            morphs = self.scene.add_entity(
                gs.morphs.Box(
                    lower = entity.lower,
                    upper = entity.upper,
                    pos = entity.pos,
                    euler = entity.euler,
                    visualization = entity.visualization,
                    collision = entity.collision,
                    )
                )
        elif entity.morphs_type == "CYLINDER":
            morphs = self.scene.add_entity(
                gs.morphs.Cylinder(
                    radius = entity.radius,
                    height = entity.height,
                    pos = entity.pos,
                    euler = entity.euler,
                    visualization = entity.visualization,
                    collision = entity.collision,
                    )
                )
        elif entity.morphs_type == "SPHERE":
            morphs = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius = entity.radius,
                    pos = entity.pos,
                    euler = entity.euler,
                    visualization = entity.visualization,
                    collision = entity.collision,
                    )
                )
        else:
            raise ValueError(f"{entity.morphs_type} not in {entity.types}")
        self.entities[name] = morphs


    def add_entities(self, entity_dict: Dict[str, Entity]) -> None:
        """
        Adds multiple morphological entities to the current Genesis scene.

        Parameters
        ----------
        entity_dict : Dict[str, Entity]
            A dictionary mapping entity names to Entity configuration objects.
        """
        for name, entity in entity_dict.items():
            self.add_entity(name, entity)


    def build(self) -> None:
        """
        Compiles and builds the underlying physics scene.
        """
        self.scene.add_entity(gs.morphs.Plane())  # add in floor plane
        self.scene.build()


    def step(self) -> None:
        """
        Advances the physical simulation forward by a single discrete time step.
        """
        self.scene.step()


    def end(self) -> None:
        """
        Safely destroys the Genesis environment and cleans up backend resources.
        """
        gs.destroy()


class ThreeJS:
    """
    Manages the configuration, generation, and live streaming of a Three.js HTML viewer.

    Parameters
    ----------
    scene_config : Dict[str, Any] | None, optional
        A dictionary of configuration parameters to inject into the JavaScript
        environment, containing the entity layout. Default is None.
    template_name : str, optional
        The filename of the HTML Jinja2 template to read from. Default is "viewer_template.html".
    output_name : str, optional
        The filename of the compiled HTML file to save. Default is "index.html".
    working_directory : str | None, optional
        The absolute path to the directory where HTML and assets reside. Default is None.
    http_port : int, optional
        The port number for the HTTP server. Default is 0 (auto-assign).
    ws_port : int, optional
        The port number for the WebSocket server. Default is 0 (auto-assign).
    """

    def __init__(
            self,
            scene_config: Optional[Dict[str, Any]] = None,
            template_name: str = "viewer_template.html",
            output_name: str = "index.html",
            working_directory: str = None,
            http_port: int = 0,
            ws_port: int = 0,
            ) -> None:
        """
        Initialises the ThreeJS live viewer manager.
        """
        if scene_config is None:
            scene_config = {}

        # 1. Package Directory: Where this python file and the Jinja template permanently live
        self.package_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Working Directory: Where the user ran the script, and where the assets/index.html should be
        if working_directory is None:
            self.working_directory = os.getcwd()
        else:
            self.working_directory = working_directory

        self.template_name = template_name
        self.output_name = output_name
        self.http_port = http_port

        self.ws_port = ws_port if ws_port != 0 else self._get_free_port()
        scene_config["ws_port"] = self.ws_port
        self.scene_config = scene_config

        # 3. Path Routing: Pull template from source, push index.html to the user's local folder
        self.template_path = os.path.join(self.package_dir, self.template_name)
        self.output_path = os.path.join(self.working_directory, self.output_name)

        self._build_html()


    def _get_free_port(self) -> int:
        """
        Asks the operating system for an available, unused port.

        Returns
        -------
        int
            An available network port number dynamically assigned by the OS.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


    def _build_html(self) -> None:
        """
        Compiles the HTML viewer from the Jinja2 template and writes it to disk.
        """
        with open(self.template_path, "r") as f:
            template = Template(f.read())

        rendered_html = template.render(scene_config=self.scene_config)

        with open(self.output_path, "w") as f:
            f.write(rendered_html)
        print(f"Successfully generated viewer at: {self.output_path}")


    def _start_local_server_and_open(self) -> http.server.HTTPServer:
        """
        Starts the background HTTP server to serve assets safely.

        Returns
        -------
        http.server.HTTPServer
            The instantiated and active HTTP server.
        """
        # Use functools.partial to cleanly specify the serving directory
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=self.working_directory)

        class ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        httpd = ReusableTCPServer(("", self.http_port), handler)
        actual_port = httpd.server_address[1]

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        print(f"HTTP Server running at http://localhost:{actual_port}")
        time.sleep(0.5)

        target_url = f"http://localhost:{actual_port}/{self.output_name}"
        webbrowser.open(target_url)

        return httpd


    def _start_websocket_server(self) -> None:
        """
        Starts an asynchronous WebSocket server in a background thread.
        """
        self.connected_clients = set()
        self.ws_loop = asyncio.new_event_loop()

        async def handler(websocket):
            self.connected_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.connected_clients.remove(websocket)

        async def ws_main():
            self.stop_future = self.ws_loop.create_future()
            async with websockets.serve(handler, "localhost", self.ws_port):
                await self.stop_future

        def run_async_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(ws_main())

        ws_thread = threading.Thread(target=run_async_loop, args=(self.ws_loop,), daemon=True)
        ws_thread.start()

        print(f"WebSocket Server running at ws://localhost:{self.ws_port}")


    def update_viewer(self, state_dict: Dict[str, Any]) -> None:
        """
        Safely broadcasts a Python dictionary payload to the Three.js viewer.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            The JSON-compatible dictionary payload to send containing live 
            transformations and scene updates.
        """
        if not hasattr(self, 'connected_clients') or not self.connected_clients:
            return

        payload = json.dumps(state_dict)

        async def broadcast():
            websockets.broadcast(self.connected_clients, payload)

        asyncio.run_coroutine_threadsafe(broadcast(), self.ws_loop)


    def run_static_html_server(self) -> None:
        """
        Executes both servers and keeps the main thread alive indefinitely.
        """
        http_server = self._start_local_server_and_open()
        self._start_websocket_server()

        try:
            print("\nServers running. Press Ctrl+C to stop and exit.\n")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down servers...")
            http_server.shutdown()
            if hasattr(self, 'stop_future') and not self.stop_future.done():
                self.ws_loop.call_soon_threadsafe(self.stop_future.set_result, True)


class Bridge(object):
    """
    Manages the coupling between Genesis physics entities and Three.js visualization payloads.
    Encapsulates the ThreeJS server manager internally for streamlined execution.

    Parameters
    ----------
    genesis : GenesisSim
        The instantiated Genesis physics simulation manager.
    entity_dict : Dict[str, Entity]
        A dictionary mapping entity string names to their `Entity` configuration objects.
    template_name : str, optional
        The filename of the HTML Jinja2 template to read from. Default is "viewer_template.html".
    output_name : str, optional
        The filename of the compiled HTML file to save. Default is "index.html".
    working_directory : str | None, optional
        The absolute path to the directory where HTML and VRM files reside. Default is None.
    http_port : int, optional
        The port number for the HTTP server. Default is 0 (auto-assign).
    ws_port : int, optional
        The port number for the WebSocket server. Default is 0 (auto-assign).
    """
    def __init__(
            self,
            genesis: GenesisSim,
            entity_dict: Dict[str, Entity],
            template_name: str = "viewer_template.html",
            output_name: str = "index.html",
            working_directory: str | None = None,
            http_port: int = 0,
            ws_port: int = 0,
            ) -> None:
        """
        Initialises the Bridge interface between Genesis and ThreeJS.
        """
        self.genesis = genesis
        self.entity_dict = entity_dict

        # Filter MJCF entities to dynamically build the internal vrm_models dictionary
        self.vrm_models = {}
        for name, ent in entity_dict.items():
            if ent.morphs_type == "MJCF" and ent.vrm_obj is not None:
                self.vrm_models[name] = ent.vrm_obj

        # Dynamically build the ThreeJS Configuration dictionary from the Entity dictionary
        threejs_config = {"entities": {}}
        for name, ent in entity_dict.items():
            if ent.morphs_type == "MJCF" and ent.vrm_file:
                threejs_config["entities"][name] = {"type": "VRM", "file": ent.vrm_file}
            elif ent.morphs_type == "BOX":
                size = [
                    ent.upper[0] - ent.lower[0],
                    ent.upper[1] - ent.lower[1],
                    ent.upper[2] - ent.lower[2]
                ]
                threejs_config["entities"][name] = {"type": "BOX", "size": size}
            elif ent.morphs_type == "CYLINDER":
                threejs_config["entities"][name] = {"type": "CYLINDER", "radius": ent.radius, "height": ent.height}
            elif ent.morphs_type == "SPHERE":
                threejs_config["entities"][name] = {"type": "SPHERE", "radius": ent.radius}
            elif ent.morphs_type == "MESH" and ent.file:
                threejs_config["entities"][name] = {"type": "MESH", "file": ent.file, "scale": ent.scale}

        # Initialize the ThreeJS Manager privately within the Bridge class
        self.viewer = ThreeJS(
            scene_config=threejs_config,
            template_name=template_name,
            output_name=output_name,
            working_directory=working_directory,
            http_port=http_port,
            ws_port=ws_port
        )

        self.compat_types = ["MJCF", "MESH", "BOX", "CYLINDER", "SPHERE"]
        self.mjcf_root_dof = 7

        self.morph_strs: Dict[str, str] = {}
        self.types: Dict[str, str] = {}
        self.joint_names: Dict[str, List[str]] = {}
        self.vrm_bone_map: Dict[str, str] = {}
        self.vrm_base_offsets: Dict[str, np.ndarray] = {}
        self.qpos: Dict[str, np.ndarray] = {}

        self.flip_rot = R.from_euler('x', -90, degrees=True)

        self._get_types()
        self._get_joint_names()
        self._get_bone_map()
        self._get_base_offsets()


    def start(self) -> None:
        """
        Starts the background web servers, launches the browser viewer, and waits
        for a WebSocket connection to be established.
        """
        self.viewer._start_local_server_and_open()
        self.viewer._start_websocket_server()
        print("Waiting 3 seconds for browser to connect...")
        time.sleep(3.0)
        print("Starting simulation loop...")


    def _get_types(self) -> None:
        """
        Populates mapping of entity names to their geometric type.
        """
        for name, entity in self.genesis.entities.items():
            morph_str = str(type(entity.morph)).upper()
            self.morph_strs[name] = morph_str

            matched_type = "UNKNOWN"
            for compat in self.compat_types:
                if compat in morph_str:
                    matched_type = compat
                    break
            self.types[name] = matched_type


    def _get_joint_names(self) -> None:
        """
        Populates list of Genesis joints for MJCF articulated models.
        """
        for name, entity_type in self.types.items():
            if entity_type == "MJCF" and name in self.vrm_models:
                entity = self.genesis.entities[name]
                genesis_joints = [joint.name for joint in entity.joints if joint.name != "root_freejoint"]
                self.joint_names[name] = genesis_joints
            else:
                self.joint_names[name] = []


    def _get_bone_map(self) -> None:
        """
        Initializes the standard VRoid to VRM humanoid bone mapping.
        """
        self.vrm_bone_map = {
            'J_Bip_C_Hips': 'hips', 'J_Bip_C_Spine': 'spine', 'J_Bip_C_Chest': 'chest',
            'J_Bip_C_UpperChest': 'upperChest', 'J_Bip_C_Neck': 'neck', 'J_Bip_C_Head': 'head',
            'J_Adj_L_FaceEye': 'leftEye', 'J_Adj_R_FaceEye': 'rightEye',
            'J_Bip_L_Shoulder': 'leftShoulder', 'J_Bip_L_UpperArm': 'leftUpperArm',
            'J_Bip_L_LowerArm': 'leftLowerArm', 'J_Bip_L_Hand': 'leftHand',
            'J_Bip_R_Shoulder': 'rightShoulder', 'J_Bip_R_UpperArm': 'rightUpperArm',
            'J_Bip_R_LowerArm': 'rightLowerArm', 'J_Bip_R_Hand': 'rightHand',
            'J_Bip_L_UpperLeg': 'leftUpperLeg', 'J_Bip_L_LowerLeg': 'leftLowerLeg',
            'J_Bip_L_Foot': 'leftFoot', 'J_Bip_L_ToeBase': 'leftToes',
            'J_Bip_R_UpperLeg': 'rightUpperLeg', 'J_Bip_R_LowerLeg': 'rightLowerLeg',
            'J_Bip_R_Foot': 'rightFoot', 'J_Bip_R_ToeBase': 'rightToes',
            'J_Bip_L_Thumb1': 'leftThumbMetacarpal', 'J_Bip_L_Thumb2': 'leftThumbProximal',
            'J_Bip_L_Thumb3': 'leftThumbDistal', 'J_Bip_L_Index1': 'leftIndexProximal',
            'J_Bip_L_Index2': 'leftIndexIntermediate', 'J_Bip_L_Index3': 'leftIndexDistal',
            'J_Bip_L_Middle1': 'leftMiddleProximal', 'J_Bip_L_Middle2': 'leftMiddleIntermediate',
            'J_Bip_L_Middle3': 'leftMiddleDistal', 'J_Bip_L_Ring1': 'leftRingProximal',
            'J_Bip_L_Ring2': 'leftRingIntermediate', 'J_Bip_L_Ring3': 'leftRingDistal',
            'J_Bip_L_Little1': 'leftLittleProximal', 'J_Bip_L_Little2': 'leftLittleIntermediate',
            'J_Bip_L_Little3': 'leftLittleDistal', 'J_Bip_R_Thumb1': 'rightThumbMetacarpal',
            'J_Bip_R_Thumb2': 'rightThumbProximal', 'J_Bip_R_Thumb3': 'rightThumbDistal',
            'J_Bip_R_Index1': 'rightIndexProximal', 'J_Bip_R_Index2': 'rightIndexIntermediate',
            'J_Bip_R_Index3': 'rightIndexDistal', 'J_Bip_R_Middle1': 'rightMiddleProximal',
            'J_Bip_R_Middle2': 'rightMiddleIntermediate', 'J_Bip_R_Middle3': 'rightMiddleDistal',
            'J_Bip_R_Ring1': 'rightRingProximal', 'J_Bip_R_Ring2': 'rightRingIntermediate',
            'J_Bip_R_Ring3': 'rightRingDistal', 'J_Bip_R_Little1': 'rightLittleProximal',
            'J_Bip_R_Little2': 'rightLittleIntermediate', 'J_Bip_R_Little3': 'rightLittleDistal',
        }


    def _get_base_offsets(self) -> None:
        """
        Calculates the local vector offset from the VRM origin to the physics root.
        """
        for name, entity_type in self.types.items():
            if entity_type == "MJCF" and name in self.vrm_models:
                vrm = self.vrm_models[name]
                names = vrm.skins_data[0]['names']
                joints = vrm.skins_data[0]['global_joints']

                hips_name = None
                for raw, std in self.vrm_bone_map.items():
                    if std == 'hips':
                        hips_name = raw
                        break

                if hips_name in names:
                    hips_idx = names.index(hips_name)
                    self.vrm_base_offsets[name] = np.array(joints[hips_idx])
                else:
                    self.vrm_base_offsets[name] = np.array([0.0, 0.0, 0.0])
            else:
                self.vrm_base_offsets[name] = np.array([0.0, 0.0, 0.0])


    def update(self, extra_data: Dict[str, Any] | None = None) -> None:
        """
        Fetches the latest qpos tensor for all entities, builds the scene payload,
        and seamlessly broadcasts it to the ThreeJS viewer.

        Parameters
        ----------
        extra_data : Dict[str, Any] | None, optional
            A dictionary containing additional scene configuration parameters
            to merge into the payload (e.g., expressions, lighting). Default is None.
        """
        for name, entity in self.genesis.entities.items():
            raw_qpos = entity.get_qpos()
            if isinstance(raw_qpos, torch.Tensor):
                self.qpos[name] = raw_qpos.detach().cpu().numpy()
            else:
                self.qpos[name] = np.array(raw_qpos)

        payload = self._get_payload(extra_data)
        self.viewer.update_viewer(payload)


    def _get_payload(
            self,
            extra_data: Dict[str, Any] | None = None,
            ) -> Dict[str, Any]:
        """
        Parses current qpos dictionaries into the Three.js coordinate space payload.

        Parameters
        ----------
        extra_data : Dict[str, Any] | None, optional
            A dictionary containing additional scene configuration parameters.
            Default is None.

        Returns
        -------
        master_payload : Dict[str, Any]
            The compiled state of the physics scene.
        """
        payload_entities = {}

        for name, qpos in self.qpos.items():
            entity_type = self.types[name]

            gx, gy, gz = qpos[0], qpos[1], qpos[2]
            world_hips_pos = np.array([gx, gz, -gy])

            gw, gqx, gqy, gqz = qpos[3], qpos[4], qpos[5], qpos[6]
            scipy_quat = [gqx, gqy, gqz, gw]
            gen_rot = R.from_quat(scipy_quat)
            final_rot = self.flip_rot * gen_rot

            if name in self.vrm_base_offsets:
                local_hips = self.vrm_base_offsets[name]
                rotated_offset = final_rot.apply(local_hips)
                world_scene_pos = world_hips_pos - rotated_offset
            else:
                world_scene_pos = world_hips_pos

            root_pos = [float(x) for x in world_scene_pos]
            root_rot = [float(x) for x in final_rot.as_quat()]

            entity_data = {
                "type": entity_type,
                "root": {
                    "position": root_pos,
                    "rotation": root_rot
                }
            }

            if entity_type == "MJCF" and name in self.vrm_models:
                bone_dict = {}
                q_idx = self.mjcf_root_dof

                for raw_joint_name in self.joint_names[name]:
                    if q_idx + 4 > len(qpos):
                        break

                    bw, bx, by, bz = qpos[q_idx:q_idx+4]
                    if raw_joint_name in self.vrm_bone_map:
                        standard_name = self.vrm_bone_map[raw_joint_name]
                        bone_dict[standard_name] = [float(bx), float(by), float(bz), float(bw)]
                    q_idx += 4

                entity_data["bones"] = bone_dict

            payload_entities[name] = entity_data

        master_payload = {"entities": payload_entities}

        if extra_data:
            if "entity_extras" in extra_data:
                for ent_name, extras in extra_data["entity_extras"].items():
                    if ent_name in master_payload["entities"]:
                        master_payload["entities"][ent_name].update(extras)
            for key, value in extra_data.items():
                if key != "entity_extras":
                    master_payload[key] = value

        return master_payload
