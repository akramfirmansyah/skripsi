#!/bin/bash

echo "Starting setup!"

NODE_RED_DIR="node-red"
INFLUXDB_DIR="influxdb"

# Check if directory node-red exists
if [ -d "$NODE_RED_DIR" ]; then
  echo "Directory '$NODE_RED_DIR' already exists."
else
  # Create directory node-red
  mkdir -p "$NODE_RED_DIR"
  echo "Directory '$NODE_RED_DIR' created."
fi

# Check if directory influxdb exists
if [ -d "$INFLUXDB_DIR" ]; then
  echo "Directory '$INFLUXDB_DIR' already exists."
else
  # Create directory influxdb
  mkdir -p "$INFLUXDB_DIR"
  echo "Directory '$INFLUXDB_DIR' created."
fi

if [ -e ./node-red/autoCapture.py ]; then
  echo "Auto Capture Code was exists!"
else
  cp ./autoCapture.py ./node-red
fi

# Start docker compose
docker compose up -d

echo "Setup completed!"
