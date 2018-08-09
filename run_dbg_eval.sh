#!/bin/bash

xvfb-run -a -s "-ac -screen 0 1280x1024x24" python dbg_evaluate.py --env_type indoor --env_name pointgoal_suncg_se --use_pixel_change False --checkpoint_dir /home/borsuvas/project/unreal/checkpoints_a3c_mattrep_small_pointgoal/
