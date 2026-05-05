"""
PBSM Bridge Communication Protocol & Reference Guide

The `Bridge` class acts as the middleman between the Genesis physics simulation 
and the Three.js WebGL renderer. It serializes the state of all objects in the 
`entity_dict` into JSON and broadcasts it at the target FPS.

=========================================
1. WHAT CAN BE CONTROLLED VIA THREE.JS?
=========================================
Three.js consumes the following properties from the JSON payload:
    - root.position     : [x, y, z] (Global translation)
    - root.rotation     : [x, y, z, w] (Global quaternion)
    - bones             : { "boneName": [x, y, z, w] } (Local joint rotations)
    - expressions       : { "expressionName": float 0.0-1.0 } (VRM Blendshapes)
    - lookAtTarget      : [x, y, z] (Global coordinate for the eyes/head to track)
    - camera            : { "position": [x,y,z], "lookAt": [x,y,z] }
    - windForce         : [x, y, z] (Vector applied to VRM SpringBones like hair/clothes)

=========================================
2. HOW TO CONTROL THEM FROM PYTHON
=========================================
You manipulate these values by updating the `Entity` object inside your main simulation loop.
The `Bridge` will automatically pack these updates and send them on `bridge.update()`.

EXAMPLE USAGE:
--------------
"""

import time
import math



def example_simulation_loop(sim, bridge, entity_dict):
    """
    Demonstrates how to manually inject expressions, gaze tracking, 
    and environmental factors during the live physics loop.
    """
    bot = entity_dict["vrm_model"]
    start_time = time.time()
    
    while True:
        sim.step()
        
        current_time = time.time() - start_time
        
        # ---------------------------------------------------------
        # A. Overriding Facial Expressions (VRM Blendshapes)
        # ---------------------------------------------------------
        # Standard VRM expressions: 'joy', 'angry', 'sorrow', 'fun', 'surprised', 'aa', 'ee', 'oh', 'blink'
        # Sending a value here will temporarily pause the Three.js procedural behaviors!
        
        if 5.0 < current_time < 10.0:
            # Force a smile between 5 and 10 seconds
            bot.expressions = {"joy": 1.0, "surprised": 0.2}
        else:
            # Clear expressions to return control to the Three.js procedural engine
            bot.expressions = {} 


        # ---------------------------------------------------------
        # B. Controlling the Gaze (Head / Eye Tracking)
        # ---------------------------------------------------------
        # You can make the character look at a specific point in 3D space.
        # Here, we make the character track a point moving in a circle.
        
        target_x = math.sin(current_time) * 3.0
        target_y = 1.5
        target_z = math.cos(current_time) * 3.0
        
        # Set the global coordinate for the eyes to track
        bot.lookAtTarget = [target_x, target_y, target_z]


        # ---------------------------------------------------------
        # C. Global Environment Variables (Sent via the Bridge)
        # ---------------------------------------------------------
        # You can inject custom data directly into the bridge state 
        # before calling bridge.update().
        
        # 1. Move the Three.js Camera
        bridge.custom_state["camera"] = {
            "position": [0, 2.0, 4.0 + math.sin(current_time)], 
            "lookAt": [0, 1.0, 0]
        }
        
        # 2. Simulate Wind (Affects VRM SpringBones like hair and skirts)
        wind_strength = math.sin(current_time * 5) * 0.5
        bridge.custom_state["windForce"] = [wind_strength, 0.0, -1.0]
        

        # ---------------------------------------------------------
        # D. Physics / Skeletal Modifications
        # ---------------------------------------------------------
        # Skeletal movements MUST be handled by Genesis, not Three.js.
        # If you want the character to wave its hand, you command the physics engine.
        
        # (Assuming you know the exact DOF index for the shoulder)
        # r_shoulder_idx = bot.get_dof_idx("rightShoulder_z")
        # wave_angle = math.radians(90) + math.sin(current_time * 10) * 0.5
        # bot.control_dofs_position([r_shoulder_idx], [wave_angle])
        
        # Send everything to Three.js
        bridge.update()