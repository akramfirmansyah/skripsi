#!/bin/bash

echo "Starting setup!"

NODE_RED_DIR="node-red"
MOSQUITTO_DIR="mosquitto"
MOSQUITTO_CONFIG_DIR="$MOSQUITTO_DIR/config"
MOSQUITTO_DATA_DIR="$MOSQUITTO_DIR/data"
INFLUXDB_DIR="influxdb"

# Check if directory node-red exists
if [ -d "$NODE_RED_DIR" ]; then
  echo "Directory '$NODE_RED_DIR' already exists."
else
  # Create directory node-red
  mkdir -p "$NODE_RED_DIR"
  echo "Directory '$NODE_RED_DIR' created."
fi

# Check if directory mosquitto exists
if [ -d "$MOSQUITTO_DIR" ]; then
  echo "Directory '$MOSQUITTO_DIR' already exists."
else
  # Create directory mosquitto
  mkdir -p "$MOSQUITTO_DIR"
  echo "Directory '$MOSQUITTO_DIR' created." 
fi

# Check if directory config of mosquitto exists
if [ -d "$MOSQUITTO_CONFIG_DIR" ]; then
  echo "Directory '$MOSQUITTO_CONFIG_DIR' already exists."
else
  # Create directory config of mosquitto
  mkdir -p "$MOSQUITTO_CONFIG_DIR"
    echo "Directory '$MOSQUITTO_CONFIG_DIR' created."
fi

# Check if directory data of mosquitto exists
if [ -d "$MOSQUITTO_DATA_DIR" ]; then
  echo "Directory '$MOSQUITTO_DATA_DIR' already exists."
else
  # Create directory data of mosquitto
  mkdir -p "$MOSQUITTO_DATA_DIR"
    echo "Directory '$MOSQUITTO_DATA_DIR' created."
fi

# Copy mosquitto configuration
echo "Copy mosquitto configuration"
cp ./mosquitto.config ./mosquitto/config

# Check if directory node-red exists
if [ -d "$INFLUXDB_DIR" ]; then
  echo "Directory '$INFLUXDB_DIR' already exists."
else
  # Create directory node-red
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
