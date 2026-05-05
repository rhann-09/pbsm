#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Based SMPL-X Modelling (PBSM)

Integration of Genesis VRM Physics and real-time audio streaming.

Setup Guide:
    
    1. Use LM studio local server with a Gemma 4 series model (E2B, E4B, 26B A4B, and 31B). Let the model
        be reachable at the default address. google/gemma-4-E4B at 4-bit quantisation has been tested.
        https://huggingface.co/google/gemma-4-E4B
        
    2. Apply the following system prompt to the Gemma 4 model:
        
        You are a voice-driven interactive avatar. You communicate entirely through spoken audio.
        Adhere to the following strict rules:
        1. NO MARKDOWN: Never use asterisks (*), bold (**), italics (_), or hashes (#).
        2. NO ACTIONS: Never write out stage directions or actions (e.g., do not write *smiles*, *sighs*, or [laughs]). 
        3. PRONUNCIATION: Write out numbers, symbols, and acronyms exactly as they should be spoken (e.g., write "three dollars" instead of "$3", and "A P I" instead of "API").
        4. CONVERSATIONAL: Speak naturally, concisely, and with conversational pacing.
        
    3. If the uv python package manager isn't installed already. Install uv as an independant package manager
        to venv or conda for example. https://docs.astral.sh/uv/
        
    4. Follow the install guide for speaches using uv https://speaches.ai/ . Additional steps are required to 
        setup TTS models https://speaches.ai/usage/text-to-speech/. Install the speaches-ai/Kokoro-82M-v1.0-ONNX
        model. 
        
    5. Spinning up a speaches server can be made easier by using a alias in your bashrc file e.g.:
        
            alias speaches="cd [speaches directory] && source .venv/bin/activate && uvicorn --factory --host 0.0.0.0 speaches.main:create_app"
        
    6. Ensure that you have the following line in your bashrc file as well:
            
            export SPEACHES_BASE_URL="http://localhost:8000"
            
    7. Before running the following script ensure that the lm studio local server and speaches server is activated
        before starting the script.
    
    8. Run this from the command line using ./tts_integration_example.py

"""

# %% Imports
import time
import json
import threading

# Package Imports
from pbsm.genesis_vrm import VRM, Entity, GenesisSim, Bridge
from pbsm.vrm_audio import VoiceManager, AudioBridge, lmstudio_stream


# %% Functions

def background_chat_loop(audio_bridge: AudioBridge):
    """
    Runs in a background thread to accept terminal input without 
    blocking the Genesis physics simulation loop.
    """
    # Wait until the user clicks the "Connect" button in the browser
    while len(audio_bridge.clients) == 0:
        time.sleep(0.5)
        
    print("\n[READY] Chat terminal initialized. Type your prompts below.")
    
    while True:
        try:
            user_input = input("\nPlease enter input: ")
            if user_input.strip().lower() in ["quit", "exit"]:
                print("Exiting chat loop...")
                # Exit the entire program safely
                import os
                os._exit(0)
            
            if user_input.strip():
                print("Generating response...")
                audio_bridge.stream_to_clients(lmstudio_stream(user_input))
        except EOFError:
            break


# %% Execution

if __name__ == "__main__":
    # Inputs
    file_path = "template_model.vrm"
    output_file = "vroid_model.xml"
    name = "vrm_model"
    
    target_fps = 60
    dt = 1.0 / target_fps
    
    # Initialise VoiceManager and AudioBridge with dynamic port mapping
    voice = VoiceManager()
    audio_bridge = AudioBridge(voice_manager=voice, port=0)
    audio_bridge.start()

    # Export the dynamic port to a JSON file so the HTML viewer can fetch it
    with open("audio_config.json", "w") as f:
        json.dump({"port": audio_bridge.port}, f)

    # Create MJCF xml file
    vrm_model = VRM(file_path)
    vrm_model.vrm2mjcf(output_file=output_file)
    
    # Create entities
    entity1 = Entity(
        "MJCF",
        file=output_file,
        vrm_file=file_path,
        vrm_obj=vrm_model,
        euler=(90, 0, 0),
        pos=(0.0, 0.0, 0.0)
        )
    
    entity_dict = {name: entity1}
    
    # Initialise genesis sim class
    sim = GenesisSim(
        show_viewer=False,
        logging_level="error",
        gravity=(0.0, 0.0, 0.0),
        )
    
    # Go through genesis pipeline
    sim.initialise()
    sim.create_scene()
    sim.add_entities(entity_dict)
    sim.build()
    
    # Initialise Physics Bridge
    bridge = Bridge(genesis=sim, entity_dict=entity_dict)
    
    # Start network rendering wrapper (This automatically opens the browser)
    bridge.start()

    # Launch background thread for non-blocking LLM input
    chat_thread = threading.Thread(target=background_chat_loop, args=(audio_bridge,), daemon=True)
    chat_thread.start()

    print("\n⏳ Waiting for user to click 'Connect Physics & Audio' in the browser...")

    # FREEZE PHYSICS: Wait until the HTML audio socket connects
    while len(audio_bridge.clients) == 0:
        time.sleep(0.1)
        
    print("▶️ Frontend connected! Starting simulation loop...")

    # Simulation loop
    try:
        while True:
            step_start = time.perf_counter()
            
            sim.step()
            bridge.update()
            
            step_duration = time.perf_counter() - step_start
            sleep_time = dt - step_duration
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user, terminating")
    finally:
        sim.end()
        print("Simulation successfully terminated.")