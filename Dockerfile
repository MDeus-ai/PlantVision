# Dockerfile

# --- Stage 1: Build Stage ---
# Use a full Python image to install dependencies
FROM python:3.10-slim as builder

# Set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Stage ---
# Use the same base image for the final, lean image
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the entire project context
COPY . .

# Install our project code as a package.
# This makes all `src.PlantVision...` imports work inside the container.
RUN pip install -e .

# The command to run when the container starts.
# We use the python -m convention, which is the robust way to run modules.
CMD ["python", "-m", "PlantVision.train"]