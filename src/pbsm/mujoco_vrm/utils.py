#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Utility functions for the pbsm.mujoco_smplx sub-package.
"""

# %% Imports

# Standard Library Imports
import asyncio
import http.server
import json
import os
import shutil
import socketserver
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, List
from xml.dom import minidom

# Third-Party Imports
import mujoco
import mujoco.viewer
import nest_asyncio
import networkx as nx
import numpy as np
import pygltflib
import trimesh
import websockets


# %% Classes

class VRM(object):
    """
    A class for parsing .vrm files and simulating them using MuJoCo for physics 
    and Three.js for rendering.

    * Reads .vrm files as a glTF file using pygltflib.
    * Extracts the kinematic tree and automatically converts it to a NetworkX graph.
    * Segments the body mesh and converts the vertices to convex hulls using trimesh.
    * Combines the network graph and convex hulls into a MuJoCo MJCF .xml file.
    * Runs MuJoCo and Three.js to simulate and render the .vrm file.
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

    # %% Public Functions

    def extract_vrm_skeleton(self, skin_index: int = 0) -> Tuple[List[str], nx.DiGraph, np.ndarray]:
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

    def extract_mesh_skinning_data(self, mesh_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def segment_by_dominant_joint(self,
                                  skin_index: int,
                                  vertices: np.ndarray,
                                  bone_indices: np.ndarray,
                                  bone_weights: np.ndarray) -> Dict[str, np.ndarray]:
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

    def generate_convex_hulls(self,
                              segmented_parts: Dict[str, np.ndarray],
                              density_kg_per_m3: float = 1000.0) -> Dict[str, trimesh.Trimesh]:
        """
        Generate convex hulls for each segmented part.
        
        Creates 3D convex hulls for each segmented part of the body mesh, applying 
        a density to calculate the physics inertia and mass of the body part.

        Parameters
        ----------
        segmented_parts : Dict[str, np.ndarray]
            Dictionary of segmented part names mapped to vertex pointclouds.
        density_kg_per_m3 : float, optional
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
                hull.density = density_kg_per_m3
                hulls_dict[joint_name] = hull
            except Exception as e:
                print(f"Could not generate hull for {joint_name}: {e}")

        return hulls_dict

    def generate_mjcf(self,
                      skin_index: int,
                      raw_vertices: np.ndarray,
                      hulls_dict: dict,
                      stl_folder: str = "STL",
                      output_file: str = "vrm_physics_body.xml") -> None:
        """
        Generates the pure physics-focused MuJoCo XML (MJCF) kinematic tree and exports hulls.

        Parameters
        ----------
        skin_index : int
            Index of the skin reference to construct the kinematic tree.
        raw_vertices : np.ndarray
            Array of raw vertex positions, used to calculate the physical floor offset.
        hulls_dict : dict
            Dictionary mapping joint names to `trimesh.Trimesh` convex hulls.
        stl_folder : str, optional
            Directory to export the generated STL collision files. Default is "STL".
        output_file : str, optional
            Filename for the generated MJCF XML file. Default is "vrm_physics_body.xml".

        Returns
        -------
        None
        """
        if os.path.exists(stl_folder):
            shutil.rmtree(stl_folder)
        os.mkdir(stl_folder)

        for name, hull in hulls_dict.items():
            hull.export(os.path.join(stl_folder, f"{name}.stl"))

        mujoco = ET.Element("mujoco", model="vrm_physics_body")
        ET.SubElement(mujoco, "compiler", angle="radian", meshdir=stl_folder, autolimits="true")

        default = ET.SubElement(mujoco, "default")
        ET.SubElement(default, "geom", density="1000", contype="1", conaffinity="0")
        ET.SubElement(default, "joint", damping="0.01", stiffness="0.0")

        asset = ET.SubElement(mujoco, "asset")
        for name in hulls_dict.keys():
            ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl")

        worldbody = ET.SubElement(mujoco, "worldbody")
        ET.SubElement(worldbody, "geom", type="plane", size="5 5 0.1", name="floor",
                      zaxis="0 1 0", contype="0", conaffinity="1")

        def f2s(arr):
            return f"{arr[0]:.6f} {arr[1]:.6f} {arr[2]:.6f}"

        names = self.skins_data[skin_index]['names']
        network = self.skins_data[skin_index]['network']
        joints = self.skins_data[skin_index]['global_joints']

        roots = [n for n, d in network.in_degree() if d == 0]
        root_name = roots[0]

        # Traverse past empty visual 'Root' nodes until we find the actual structural Pelvis
        while len(list(network.successors(root_name))) == 1:
            root_name = list(network.successors(root_name))[0]

        root_global_pos = joints[names.index(root_name)]

        verts_local = raw_vertices - root_global_pos
        min_y_local = np.min(verts_local[:, 1])
        spawn_pos = np.array([0.0, -min_y_local + 0.2, 0.0])

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

        xml_str = ET.tostring(mujoco, encoding='utf-8')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
        pretty_xml = os.linesep.join([s for s in pretty_xml.splitlines() if s.strip()])

        with open(output_file, "w") as f:
            f.write(pretty_xml)

    def start_physics_stream(self,
                             skin_index: int,
                             output_file: str,
                             runtime: float = 300,
                             port: int = 8765,
                             show_viewer: bool = True) -> None:
        """
        Runs the HTTP server, MuJoCo decoupled physics loop, and WebSocket stream.

        Parameters
        ----------
        skin_index : int
            Index of the skin reference to map physics joints.
        output_file : str
            Path to the compiled MuJoCo MJCF XML file to simulate.
        runtime : float, optional
            Maximum simulation duration in seconds. Default is 300.
        port : int, optional
            Local port used for the WebSocket broadcaster. Default is 8765.
        show_viewer : bool, optional
            If True, launches the native MuJoCo passive viewer alongside the headless web socket. 
            Default is True.

        Returns
        -------
        None
        """
        def serve_http():
            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass

            class ReuseTCPServer(socketserver.TCPServer):
                allow_reuse_address = True

            with ReuseTCPServer(("", 8000), QuietHandler) as httpd:
                httpd.serve_forever()

        threading.Thread(target=serve_http, daemon=True).start()
        print("\n[Server] HTTP Viewer hosted at http://localhost:8000/index.html")

        model = mujoco.MjModel.from_xml_path(output_file)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        model.opt.gravity[:] = [0, -10, 0]

        viewer = mujoco.viewer.launch_passive(model, data) if show_viewer else None

        network = self.skins_data[skin_index]['network']
        roots = [n for n, d in network.in_degree() if d == 0]
        root_name = roots[0]

        while len(list(network.successors(root_name))) == 1:
            root_name = list(network.successors(root_name))[0]

        def physics_worker():
            step_time = model.opt.timestep
            while data.time < runtime:
                if viewer and not viewer.is_running():
                    break

                step_start = time.time()
                mujoco.mj_step(model, data)

                if viewer:
                    viewer.sync()

                elapsed = time.time() - step_start
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

        threading.Thread(target=physics_worker, daemon=True).start()

        async def stream_client(websocket, path=""):
            print("[Connected] Web client joined! Streaming data...")
            try:
                while data.time < runtime:
                    if viewer and not viewer.is_running():
                        break

                    bone_rotations = {}

                    for jnt_id in range(model.njnt):
                        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
                        if not jnt_name: continue

                        jnt_type = model.jnt_type[jnt_id]
                        adr = model.jnt_qposadr[jnt_id]

                        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                            px, py, pz, qw, qx, qy, qz = data.qpos[adr : adr + 7]
                            body_id = model.jnt_bodyid[jnt_id]
                            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

                            if body_name:
                                bone_rotations[body_name] = {
                                    "pos": {"x": float(px), "y": float(py), "z": float(pz)},
                                    "rot": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)}
                                }

                        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                            qw, qx, qy, qz = data.qpos[adr : adr + 4]
                            bone_rotations[jnt_name] = {
                                "rot": {"x": float(qx), "y": float(qy), "z": float(qz), "w": float(qw)}
                            }

                    await websocket.send(json.dumps(bone_rotations))
                    await asyncio.sleep(1 / 60)
            except websockets.exceptions.ConnectionClosed:
                print("[Disconnected] Web client left.")

        print(f"[Server] Physics Stream listening on ws://localhost:{port}")
        start_server = websockets.serve(stream_client, "localhost", port)

        try:
            loop = asyncio.get_running_loop()
            nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(start_server)
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            print("\nShutdown gracefully.")

    # %% Private Functions

    def _load_and_inspect(self, file_path: str) -> pygltflib.GLTF2:
        """
        Loads the .vrm binary file using pygltflib.

        Parameters
        ----------
        file_path : str
            Path to the target VRM file.

        Returns
        -------
        gltf : pygltflib.GLTF2
            The parsed glTF object.
        """
        gltf = pygltflib.GLTF2().load_binary(file_path)
        print(f"Successfully loaded: {file_path}")
        return gltf

    def _get_accessor_data(self, accessor_idx: int) -> np.ndarray:
        """
        Reads raw binary glTF buffer data and converts it into a Numpy array.

        Parameters
        ----------
        accessor_idx : int
            Index of the accessor within the glTF file.

        Returns
        -------
        array : np.ndarray
            The formatted numeric array corresponding to the accessor data.
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