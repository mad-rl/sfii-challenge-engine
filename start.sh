#!/bin/bash

python3.7 -m retro.import $ROM_PATH

xvfb-run -s "-screen 0 640x448x24" python3.7 /sfii-challenge/mad-rl-framework/src/environments/main.py