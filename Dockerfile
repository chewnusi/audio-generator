# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-venv python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Clone the riffusion repository
RUN git clone https://github.com/riffusion/riffusion.git

# Move the application files to the correct directory
COPY app.py /app/riffusion/riffusion/streamlit/
COPY requirements.txt /app/riffusion/riffusion/streamlit/

# Change working directory to riffusion
WORKDIR /app/riffusion

# Create a virtual environment and activate it
RUN python3 -m venv venv && \
    /bin/bash -c "source venv/bin/activate" && \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install ffmpeg-python soundfile torchaudio pysoundfile && \
    pip install -r requirements_all.txt && \
    pip install -e . && \
    pip install -r riffusion/streamlit/requirements.txt

# Modify the spectrogram_converter.py file
RUN sed -i '/max_iter=params.max_mel_iters/s/^/#/' riffusion/spectrogram_converter.py && \
    sed -i '/tolerance_loss=1e-5/s/^/#/' riffusion/spectrogram_converter.py && \
    sed -i '/tolerance_change=1e-8/s/^/#/' riffusion/spectrogram_converter.py && \
    sed -i '/sgdargs=None/s/^/#/' riffusion/spectrogram_converter.py

# Expose the port Streamlit is running on
EXPOSE 8501

# Ensure the virtual environment is activated and run the Streamlit app
CMD ["/bin/bash", "-c", "source venv/bin/activate && streamlit run riffusion/streamlit/app.py"]
