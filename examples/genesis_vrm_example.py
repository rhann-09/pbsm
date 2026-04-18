#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Example script demonstrating coupling off the Three.js renderer and Genesis physics
backend with a vroid derived .vrm model
"""

# %% Imports

# Standard Library Imports
import time

# Package Imports
from pbsm.genesis_vrm import VRM, Entity, GenesisSim, Bridge

# %% Execution

# inputs
file_path = "template_model.vrm"
mesh_path = "jetpack.stl"
output_file = "vroid_model.xml"
name1 = "vrm_model1"
name2 = "vrm_model2"
box_name = "box"
cylinder_name = "cylinder"
sphere_name = "sphere"
mesh_name = "jetpack"
target_fps = 60
dt = 1.0 / target_fps
total_frames = 1000

# create MJCF xml file
vrm_model = VRM(file_path)
vrm_model.vrm2mjcf(output_file=output_file)

# create entities
entity1 = Entity("MJCF", file=output_file, vrm_file=file_path, vrm_obj=vrm_model, euler=(90, 0, 0), pos=(0.0, 0.0, 0.0))
entity2 = Entity("MJCF", file=output_file, vrm_file=file_path, vrm_obj=vrm_model, euler=(90, 0, 0), pos=(2.0, 0.0, 0.0))
entity3 = Entity("BOX", lower=(3.0, 0.0, 0.5), upper=(4.0, 1.0, 1.5))
entity4 = Entity("CYLINDER", pos=(6.0, 0.0, 1.0))
entity5 = Entity("SPHERE", pos=(8.0, 0.0, 1.0))
entity6 = Entity("MESH", file=mesh_path, pos=(10, 0.0, 0.5), scale=0.01)

# entity dictionary
entity_dict = {
    name1: entity1,
    name2: entity2,
    box_name: entity3,
    cylinder_name: entity4,
    sphere_name: entity5,
    mesh_name: entity6,
}

# initialise genesis sim class
sim = GenesisSim(show_viewer=False)
sim.initialise()
sim.create_scene()
sim.add_entities(entity_dict)
sim.build()

# initialise Bridge
bridge = Bridge(genesis=sim, entity_dict=entity_dict)

# Start network rendering wrapper
bridge.start()

# time loop
for ts in range(total_frames):
    step_start = time.perf_counter()
    
    sim.step()
    
    bridge.update()
    
    step_duration = time.perf_counter() - step_start
    sleep_time = dt - step_duration
    
    if sleep_time > 0:
        time.sleep(sleep_time)

# finish
sim.end()