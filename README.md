# NCTU-Pattern-Recognition-2023-Spring

## Environment details
1. **Docker container**
    - Download docker image from [11.0.3-cudnn8-devel-ubuntu18.04](https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.0)
    - Create a container
    
    ```bash
    docker pull nvidia/cuda:11.0.3-devel-ubuntu18.04
    docker run -d -it --gpus all -p [Port] -v [Path]:[Container Path] --name [name] [image ID]
    ```
    
2. **Execute the container and update** 
    - python3 —version: Python 3.6.9
    ```bash
    apt-get update
    apt-get upgrade -y
    apt-get install python3 -y
    apt-get install python3-pip -y
    pip3 install --upgrade pip
    ```

3. **Install some requirements module**
    
    ```bash
    pip install matplotlib
    pip install numpy
    pip install pandas
    pip install tensorflow
    pip install opencv-python
    pip install scikit-learn
    pip install pydot
    ```