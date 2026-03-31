#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Testing Utility functions for the pbsm.mujoco_smplx sub-package.
"""


# Standard Imports
from unittest.mock import patch, mock_open

# Third-Party Imports
import networkx
import pytest
import torch
import numpy as np
from scipy.spatial import cKDTree

# Package-Specific Imports
import pbsm.mujoco_smplx.utils as utils


def test_make_name_and_network():
    """Check output type and size."""
    
    names, network = utils.make_name_and_network()
    
    # test types
    assert isinstance(names, list)
    assert isinstance(names[0], str)
    assert isinstance(network, networkx.classes.graph.Graph)
    
    # test size
    assert len(names) == 127
    assert network.number_of_edges() == 67
    assert network.number_of_nodes() == 68


@pytest.fixture
def valid_params():
    """Provides a set of valid parameters for the model."""
    
    return {
        "model_path": "/path/to/model",
        "gender": "neutral",
        "ext": "npz",
        "betas": torch.zeros((1, 10), dtype=torch.float32),
        "use_pca": False,
        "flat_hand_mean": True
        }


@patch('smplx.SMPLX', autospec=True)
def test_default_smplx_model_success(mock_smplx, valid_params):
    """Test that the function returns a model when given valid inputs."""
    
    # mock_smplx is the class, mock_instance is what is returned when called
    mock_instance = mock_smplx.return_value 

    model = utils.default_smplx_model(**valid_params)

    # Verify smplx.SMPLX was called with correct arguments
    mock_smplx.assert_called_once_with(
        valid_params["model_path"],
        gender=valid_params["gender"],
        ext=valid_params["ext"],
        betas=valid_params["betas"],
        use_pca=valid_params["use_pca"],
        flat_hand_mean=valid_params["flat_hand_mean"]
    )
    assert model == mock_instance


def test_perfect_symmetry():
    """Test a set of points that are perfectly symmetric across the X-axis."""
    # Points: 
    # 0: Right side
    # 1: Left side (mirror of 0)
    # 2: Further right
    # 3: Further left (mirror of 2)
    vertices = np.array([
        [ 1.0, 2.0, 3.0],  # index 0
        [-1.0, 2.0, 3.0],  # index 1
        [ 5.0, 0.0, 0.0],  # index 2
        [-5.0, 0.0, 0.0],  # index 3
    ])
    
    expected_map = np.array([1, 0, 3, 2])
    mirror_map = utils.find_vertex_symmetry(vertices)
    
    np.testing.assert_array_equal(mirror_map, expected_map)


def test_make_symmetric_weights_success():
    """Check the weight averaging logic across vertices and joints."""
    lbs_weights = np.array([[0.8, 0.2], [0.1, 0.9]])
    v_mirror_map = np.array([1, 0])
    j_mirror_map = {0: 1, 1: 0}
    
    # Expected for index 0, joint 0: (weights[0,0] + weights[v_mirror[0], j_mirror[0]]) / 2
    # (0.8 + weights[1,1]) / 2 = (0.8 + 0.9) / 2 = 0.85
    expected = np.array([[0.85, 0.15], [0.15, 0.85]])
    result = utils.make_symmetric_weights(lbs_weights, v_mirror_map, j_mirror_map)
    np.testing.assert_array_almost_equal(result, expected)


@patch('trimesh.remesh.subdivide')
def test_subdivide_by_attributes_success(mock_subdivide):
    """Ensure subdivide is called the correct number of times and returns data."""
    v = np.zeros((3, 3))
    f = np.zeros((1, 3))
    attrs = {'weights': np.zeros(3)}
    mock_subdivide.return_value = (v, f, attrs)
    
    v_out, f_out, attrs_out = utils.subdivide_by_attributes(v, f, attrs, iterations=2)
    
    assert mock_subdivide.call_count == 2
    np.testing.assert_array_equal(v_out, v)


def test_load_aligned_smplx_uv_success():
    """Test parsing of a mock OBJ file with UV coordinates."""
    obj_content = "vt 0.1 0.2\nvt 0.3 0.4\nf 1/1/1 2/2/2 3/1/3\n"
    with patch("builtins.open", mock_open(read_data=obj_content)):
        uvs = utils.load_aligned_smplx_uv("dummy.obj", num_vertices=3)
        # v1 (idx 0) -> vt1 (0.1, 0.2)
        # v2 (idx 1) -> vt2 (0.3, 0.4)
        # v3 (idx 2) -> vt1 (0.1, 0.2)
        expected = np.array([[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]])
        np.testing.assert_array_almost_equal(uvs, expected)


def test_segment_by_provided_weights_success():
    """Verify pointcloud segmentation based on dominant LBS weights."""
    names = ["j0", "j1"]
    segment_joints = ["j0", "j1"]
    pc = np.array([[0,0,0], [1,1,1]])
    weights = np.array([[0.9, 0.1], [0.2, 0.8]]) # v0 belongs to j0, v1 belongs to j1
    
    result = utils.segment_by_provided_weights(names, segment_joints, pc, weights)
    
    assert "j0" in result
    assert "j1" in result
    np.testing.assert_array_equal(result["j0"], [[0,0,0]])
    np.testing.assert_array_equal(result["j1"], [[1,1,1]])


@patch('xml.dom.minidom.parseString')
@patch('builtins.open', new_callable=mock_open)
@patch('os.path.exists', return_value=True)
def test_generate_full_body_mjcf_smoke(mock_exists, mock_file, mock_minidom):
    """Smoke test to ensure generate_full_body_mjcf runs and writes a file."""
    names = ["pelvis", "j1"]
    network = networkx.Graph()
    network.add_edge("pelvis", "j1")
    joints = np.zeros((2, 3))
    segment_joints = ["pelvis", "j1"]
    pc = np.zeros((10, 3))
    faces = np.zeros((2, 3), dtype=int)
    weights = np.ones((10, 2))
    uvs = np.zeros((10, 2))
    
    # Mock XML pretty-printing
    mock_minidom.return_value.toprettyxml.return_value = "<mujoco/>"
    
    utils.generate_full_body_mjcf(
        network, names, joints, segment_joints, pc, faces, weights, uvs,
        output_file="test.xml"
    )
    
    mock_file.assert_called_once_with("test.xml", "w")