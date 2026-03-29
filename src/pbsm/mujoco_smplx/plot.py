#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Plottings functions for the pbsm.mujoco_smplx sub-package.
"""

# %% Imports

# Standard Imports
from typing import Dict

# Third-Party Imports
import smplx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import trimesh

pio.renderers.default = "browser"

# %% Functions

def plot_vertices(model: smplx.SMPLX) -> None:
    """
    Renders an interactive 3D scatter plot of the SMPL-X model's raw vertices.

    Parameters
    ----------
    model : smplx.SMPLX
        The instantiated SMPL-X model.

    Returns
    -------
    None
    """
    vertices = model(return_verts=True).vertices.detach().cpu().numpy().squeeze()

    vertices_scatter = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Joints',
    )

    fig = go.Figure(data=vertices_scatter)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True),
            aspectmode='data' # ensure aspect ratio is correct
        ),
    )

    fig.show()
    
    
def plot_mesh(model: smplx.SMPLX) -> None:
    """
    Renders an interactive 3D plot showing both the SMPL-X mesh and internal joints.

    Parameters
    ----------
    model : smplx.SMPLX
        The instantiated SMPL-X model.

    Returns
    -------
    None
    """
    
    output = model(return_verts=True)
    joints = output.joints.detach().cpu().numpy().squeeze()
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces

    # plotly interactive plot
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.70)

    scatter = go.Scatter3d(
        x=joints[:, 0],
        y=joints[:, 1],
        z=joints[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Joints')

    fig = go.Figure(data=[mesh, scatter])

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True),
            aspectmode='data'))

    fig.show()
    
    
def plot_segments(segmented_parts: Dict[str, np.ndarray],
                  full_pointcloud: np.ndarray = None) -> None:
    """
    Visualizes the pointcloud segments colored by their assigned body parts.

    Parameters
    ----------
    segmented_parts : Dict[str, np.ndarray]
        Dictionary mapping joint names to their segmented (V, 3) vertex arrays.
    full_pointcloud : np.ndarray, optional
        The complete (N, 3) pointcloud to render unassigned points as ghosted grey, by default None.

    Returns
    -------
    None
    """
    fig = go.Figure()

    # Use a large qualitative color palette for the different hand joints
    colors = px.colors.qualitative.Alphabet
    all_segmented_points = []

    for idx, (joint_name, vertices) in enumerate(segmented_parts.items()):
        if len(vertices) == 0:
            print(f"Warning: {joint_name} has no vertices assigned.")
            continue

        all_segmented_points.append(vertices)

        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            mode='markers',
            marker=dict(size=3, color=colors[idx % len(colors)], opacity=0.9),
            name=joint_name
        ))

    # Subtract segmented points from full pointcloud and plot the rest
    if full_pointcloud is not None and len(all_segmented_points) > 0:
        seg_stacked = np.vstack(all_segmented_points)

        # Fast 2D array set difference using contiguous memory views
        full_view = np.ascontiguousarray(full_pointcloud).view([('', full_pointcloud.dtype)] * full_pointcloud.shape[1])
        seg_view = np.ascontiguousarray(seg_stacked).view([('', seg_stacked.dtype)] * seg_stacked.shape[1])

        rest_view = np.setdiff1d(full_view, seg_view)
        rest_of_body = rest_view.view(full_pointcloud.dtype).reshape(-1, full_pointcloud.shape[1])

        fig.add_trace(go.Scatter3d(
            x=rest_of_body[:, 0], y=rest_of_body[:, 1], z=rest_of_body[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='grey', opacity=0.15),
            name="Rest of Body"
        ))

    fig.update_layout(
        title="Segmented Right Hand by LBS Weights",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    
    
def plot_collision_hulls(hulls_dict: Dict[str, trimesh.Trimesh]) -> None:
    """
    Renders an interactive 3D plot of the generated convex hulls for physics collision.

    Parameters
    ----------
    hulls_dict : Dict[str, trimesh.Trimesh]
        Dictionary mapping part names to their calculated Trimesh convex hull objects.

    Returns
    -------
    None
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Alphabet

    # Plot the Solid Convex Hulls
    for idx, (part_name, t_mesh) in enumerate(hulls_dict.items()):
        if t_mesh.is_empty:
            continue

        vertices = t_mesh.vertices
        faces = t_mesh.faces

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=part_name,
            color=colors[idx % len(colors)],
            opacity=0.85,
            flatshading=True, # Emphasizes the rigid, low-poly nature of the hull
            showlegend=True
        ))

    fig.update_layout(
        title="Physics Collision Proxies (Convex Hulls) & Body Pointcloud",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()