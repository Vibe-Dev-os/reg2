#!/bin/bash

# Start script for Render deployment
echo "Starting Student Grade Predictor API..."

# Run uvicorn with host 0.0.0.0 and port from environment
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
