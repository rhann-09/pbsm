#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Testing Utility functions for the pbsm.mujoco_smplx sub-package.
"""


# Standard Imports
from unittest.mock import patch

# Third-Party Imports
import networkx
import pytest
import torch
import numpy as np
from scipy.spatial import cKDTree



# Package-Specific Imports
import pbsm.mujoco_smplx.utils as utils


# %% make_name_and_network

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

# %% default_smplx_model

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

@patch('smplx.SMPLX')
def test_betas_conversion_to_float32(mock_smplx, valid_params):
    """Test that betas are automatically converted to float32 if passed as float64."""
    
    valid_params["betas"] = torch.zeros((1, 10), dtype=torch.float64)
    
    utils.default_smplx_model(**valid_params)
    
    # Get the 'betas' argument from the call to SMPLX
    args, kwargs = mock_smplx.call_args
    passed_betas = kwargs['betas']
    
    assert passed_betas.dtype == torch.float32

@pytest.mark.parametrize("key, bad_value, expected_msg",
    [
        ("model_path", 123, "model_path must be a str"),
        ("gender", ["male"], "gender must be a str"),
        ("ext", None, "ext must be a str"),
        ("use_pca", "True", "use_pca must be a bool"),
        ("flat_hand_mean", 1, "flat_hand_mean must be a bool"),
        ("betas", [0.1, 0.2], "betas must be a torch.Tensor"),
    ])
def test_type_errors(key, bad_value, expected_msg, valid_params):
    """Test that TypeError is raised for various invalid input types."""
    
    valid_params[key] = bad_value
    
    with pytest.raises(TypeError) as excinfo:
        utils.default_smplx_model(**valid_params)
    
    assert str(excinfo.value) == expected_msg

def test_missing_required_arguments():
    """Test that omitting required arguments raises a standard Python TypeError."""
    
    with pytest.raises(TypeError):
        
        # Missing betas and other required positional/keyword args
        utils.default_smplx_model(model_path="path", gender="male", ext="pkl")
        
# %% find_vertex_symmetry

def test_find_vertex_symmetry_type():
    """Check output type and size."""
    
    try:
        utils.find_vertex_symmetry([0,0,0,0])
    except TypeError:
        assert True
    else:
        assert False

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


def test_points_on_symmetry_plane():
    """Test points that lie exactly on the YZ plane (X=0). They should map to themselves."""
    vertices = np.array([
        [0.0,  5.0,  2.0], # index 0
        [0.0, -1.0, -1.0], # index 1
        [1.0,  0.0,  0.0], # index 2 (Right)
        [-1.0, 0.0,  0.0], # index 3 (Left)
    ])
    
    # 0 and 1 map to themselves. 2 and 3 map to each other.
    expected_map = np.array([0, 1, 3, 2])
    mirror_map = utils.find_vertex_symmetry(vertices)
    
    np.testing.assert_array_equal(mirror_map, expected_map)


def test_single_vertex():
    """Test the edge case of a single vertex."""
    vertices = np.array([[3.0, 4.0, 5.0]])
    
    # Even though its true mirror doesn't exist, the KDTree will map it 
    # to the closest available point, which is itself (index 0).
    expected_map = np.array([0])
    mirror_map = utils.find_vertex_symmetry(vertices)
    
    np.testing.assert_array_equal(mirror_map, expected_map)


def test_asymmetric_closest_match():
    """Test that points map to the *closest* spatial neighbor if a perfect mirror doesn't exist."""
    vertices = np.array([
        [ 2.0, 0.0, 0.0], # index 0. Mirrored -> [-2, 0, 0]
        [-2.5, 0.0, 0.0], # index 1. Closest to [-2, 0, 0]
        [-1.0, 0.0, 0.0], # index 2. Further from [-2, 0, 0]
    ])
    
    mirror_map = utils.find_vertex_symmetry(vertices)
    
    # The mirrored version of index 0 is [-2.0, 0.0, 0.0].
    # The closest point in the original array to this mirrored coordinate is index 1.
    assert mirror_map[0] == 1
    
