# Use the official Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by libraries (e.g., gmsh) and git for porepy
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgomp1 \
    libglu1-mesa \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire contents of the current directory to the container
COPY . /app

# Install required Python packages
# porepy is installed from GitHub, others from PyPI
RUN pip install --no-cache-dir git+https://github.com/pmgbergen/porepy.git numpy scipy matplotlib gmsh

# Set the default command (displays help)
# To run a specific model, override the command, e.g.:
# docker run <image_name> python run_simulation.py 2D
CMD ["python", "run_simulation.py"]
