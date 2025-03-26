#!/bin/bash

for CONFIG_PATH in configs/Kimera/Outdoor/*.yaml
do
    echo "Running SLAM with config: $CONFIG_PATH"
    python run_slam.py "$CONFIG_PATH"
done

for CONFIG_PATH in configs/Kimera/Tunnels/*.yaml
do
    echo "Running SLAM with config: $CONFIG_PATH"
    python run_slam.py "$CONFIG_PATH"
done

for CONFIG_PATH in configs/Kimera/Hybrid/*.yaml
do
    echo "Running SLAM with config: $CONFIG_PATH"
    python run_slam.py "$CONFIG_PATH"
done