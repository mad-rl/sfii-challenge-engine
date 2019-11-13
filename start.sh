#!/bin/bash

python3.7 -m retro.import $ROM_PATH

python3.7 -m pip install -r /requirements/requirements.txt

OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=. python3.7 src/environments/main.py