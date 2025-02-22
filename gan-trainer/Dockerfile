FROM tensorflow/tensorflow:latest-jupyter

WORKDIR /app

COPY . /app 
# Copies all files in the current directory to /app inside the container

# Upgrade pip within the Docker image
RUN pip install --upgrade pip

# Install project dependencies from requirements.txt using --user flag
RUN pip install --user -r requirements.txt


RUN python -c "import tensorflow as tf; import keras; print('TensorFlow Version:', tf.__version__); print('Keras Version:', keras.__version__ if 'keras' in locals() else tf.keras.__version__)"

# Expose any ports if your application serves an interface (not strictly needed for this training script)
EXPOSE 8888

# CMD cat /proc/cpuinfo | grep -i mips


CMD python kerasGANv9.py --xDim=64 --yDim=64 --batchSize=4 --noiseSize=4 --outputDir=/app/generatedImgsv2 --trainingDir=/app/dataset

# CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"] # If you want to run a Jupyter notebook