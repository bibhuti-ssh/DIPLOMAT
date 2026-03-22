# Use Miniconda as the base image since requirements are exported conda environments
FROM continuumio/miniconda3:latest

# Install base utilities and git (required for some python packages like peft/llamafactory)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files first to leverage Docker cache
COPY requirements.txt .
COPY requirements_phase3.txt .

# Create the Data Generation (Phase 1 & 2) conda environment
# The exact hashes in requirements.txt (e.g., _libgcc_mutex=0.1=main) are Linux-64 native, 
# which perfectly matches this container architecture.
RUN conda create --name diplomat --file requirements.txt

# Create the Training (Phase 3) conda environment and inject llamafactory
RUN conda create --name diplomat_phase3 --file requirements_phase3.txt && \
    conda run -n diplomat_phase3 pip install llamafactory

# Setup Entrypoint to seamlessly use conda environments
# By default, runs the standard pipeline in the `diplomat` environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "diplomat"]
CMD ["bash", "run_full_pipeline.sh"]
