FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install essential packages
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python and Jupyter
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Install Jupyter and other dependencies
RUN pip install jupyter gwpy pandas

# Install additional system dependencies for GWpy
RUN apt-get update && apt-get install -y \
    libfftw3-dev \
    libhdf5-dev \
    libgsl-dev

# Install PyTorch
RUN pip install torch==1.13.0

# Install scikit-learn
RUN pip install scikit-learn


# Set the working directory
WORKDIR /app

# Copy the notebook into the container
COPY interTwin_wp_4.4_synthetic_data.ipynb /app/

# Expose the Jupyter notebook port
EXPOSE 8888

# Command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
