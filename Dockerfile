FROM ubuntu:22.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ./iScience_GlioLymph_classification ./iScience_GlioLymph_classification
COPY ./sample_data ./inputdata
COPY ./sample_data ./inputdata

# Set working directory
WORKDIR /iScience_GlioLymph_classification

# Create required directories and set permissions
RUN mkdir -p /outputdata /ckpts && chmod -R 755 /outputdata /ckpts

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && pip install -r /iScience_GlioLymph_classification/requirements.txt

# Define default runtime behavior
#ENTRYPOINT ["/bin/bash", "-l", "-c"]
ENTRYPOINT ["python3", "main.py"]
CMD ["--batch_size=1", "--device=cpu"]