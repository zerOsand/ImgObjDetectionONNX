# Image Object Detection

![](screenshots/cmd_example.png)

### Installation

This project uses `pipenv` as its virtualenv management tool, which you will need to have installed:

1. Install pipenv
```bash
pip install pipenv
```
2. You will need to make sure you have `tkinter` installed. This can be done as the following according to your OS:

```bash
# Check if tkinter is installed:
python -m tkinter

# Otherwise, install it

# UBUNTU / DEBIAN
sudo apt-get install python3-tk

# MacOS
brew install python-tk@3.10

# Fedora
sudo dnf install python3-tkinter

# CentOS
sudo yum install python3-tkinter

# Make sure to specify correct Python version:
sudo apt-get install python3.10-tk
```

Now, just run the following command to activate the env and install all the dependencies

3. Activate the virtual environment
```bash
pipenv shell
```

4. Install dependencies
```bash
pipenv install
```

5. Downloading the models

We want to download the models and store them inside `ml/models/`. To achieve this, you can do the following:
```bash
cd ml
mkdir models
cd models
# Download retina-net (size = 130 mb)
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/retinanet_resnet50_fpn_coco-eeacb38b.pth

# Go back to the project directory
cd ../..
```

### Starting the server

Ensure you are in the project directory and then simply: 
```bash
python -m backend.model_server
```

If you get an import error when running the server, you might need to do the following:
```bash
export PYTHONPATH=pwd # path to this project directory
```
or if you are using Windows:
```pwsh
set PYTHONPATH=%PYTHONPATH%;C:\project_path
```

### Command line tool

For the command line tool, you will need to specify the path to your input image, the path & name that you want for your output image & output csv and select one from the following models: retina-net, yolov3, tiny-yolov3.

Simply run:
```bash
python cmd_interface.py detect -h 
```
to learn how to pass in the arguments.

Here is an example command:
```bash
python cmd_interface.py detect --input_path {path_to_input_img} --output_img {path_to_output_img} --output_csv {path_to_output_csv} --model_path {path_to_ONNX_model}
```

There are sample images present in the `input_images` directory and a `output_images` directory where you can store your results as a way to test out this project.

### ONNX Export Process
In the file `pth2onnx.py`, I converted the `retinanet_resnet50_fpn_coco-eeacb38b.pth` model into an ONNX model.

To achieve this, I borrowed insights from the [ImageAI](https://github.com/OlafenwaMoses/ImageAI/tree/master) repository while using the code provided in the [image_object_detection](https://github.com/Shreneken/image_object_detection) repository.

The `pth` file is loaded into a PyTorch model. Then, (in previous versions), I took a sample image and preprocessed it according to the RetinaNet input restraints. However, there were many issues with this, and so I ended up creating a `dummy_input` variable that creates a random `(1, 3, 800, 1333)` Torch Tensor (batch, channels, height, width). 

Unfortunately, the model does poorly on images with a longer height compared to a longer width due to the ONNX input limitations that expect a fixed input size. With and without dynamic axes, this issue still was very difficult to find a solution to. However, I do plan to fix this issue.

Even with Tensors such as `(1, 3, 1333, 800)`, the results were the same. I have an inkling the issue lies in the preprocessing steps and will investigate further.