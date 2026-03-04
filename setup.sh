#!/bin/bash
# Install additional libraries
apt-get update
apt-get install -y libgl1-mesa-glx libgl1-mesa-dri libgl1 libglx0 || true
ldconfig