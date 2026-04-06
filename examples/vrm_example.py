#!/home/mango/anaconda3/envs/ai/bin/python
# -*- coding: utf-8 -*-
"""
Date: Mon Apr  6 16:16:23 2026
Author: Reuben H. (mango)

Project:
    Physics-Based SMPL-X Modelling (PBSM)

"""
import pbsm.main

in_file = "template_model.vrm"
out_file = "vroid_model.xml"

vrm_model = pbsm.main.vrm2mjcf(in_file, output_file=out_file)
pbsm.main.vrm_sim(out_file, vrm_model)