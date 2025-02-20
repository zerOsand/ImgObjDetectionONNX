import torch
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch.onnx
from imageai import Detection
import os
import uuid
from typing import Union
import numpy as np
from PIL import Image
import cv2
from torchvision.io import ImageReadMode, image
import torchvision.transforms.v2 as T


def read_file(path: str) -> torch.Tensor:
    data = image.read_file(path)
    return data


def decode_image(
    input: torch.Tensor, mode: ImageReadMode = ImageReadMode.UNCHANGED
) -> torch.Tensor:
    output = image.decode_image(input, mode)
    return output


def read_image(
    path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED
) -> torch.Tensor:
    data = read_file(path)
    return decode_image(data, mode)


def save_temp_img(input_image: Union[np.ndarray, Image.Image]) -> str:

    temp_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"{str(uuid.uuid4())}.jpg"
    )
    if type(input_image) == np.ndarray:
        cv2.imwrite(temp_path, input_image)
    elif isinstance(input_image, Image.Image):
        input_image.save(temp_path)
    else:
        raise ValueError(
            f"Invalid image input. Supported formats are OpenCV/Numpy array, PIL image or image file path"
        )

    return temp_path


def apply_transforms(image, resolution=800):
    _transform = T.Compose(
        [
            T.Resize(
                resolution + resolution // 8, interpolation=T.InterpolationMode.BILINEAR
            ),
            T.CenterCrop(resolution),
            T.ToTensor(),
        ]
    )
    return _transform(image)


def load_image(input_image):
    """
    Loads image from the given path.
    """
    allowed_file_extensions = ["jpg", "jpeg", "png"]
    images = []
    scaled_images = []
    fnames = []

    delete_file = False
    if type(input_image) is not str:
        input_image = save_temp_img(input_image=input_image)
        delete_file = True

    if os.path.isfile(input_image):
        if input_image.rsplit(".")[-1].lower() in allowed_file_extensions:
            img = read_image(input_image, ImageReadMode.RGB)
            img = apply_transforms(img)

            images.append(img)
            scaled_images.append(img.div(255.0).to(device))
            fnames.append(os.path.basename(input_image))
    else:
        raise ValueError(f"Input image with path {input_image} not a valid file")

    if delete_file:
        os.remove(input_image)

    if images:
        return (fnames, images, scaled_images)
    raise RuntimeError(
        f"Error loading image from input."
        "\nEnsure the folder contains images,"
        " allowed file extensions are .jpg, .jpeg, .png"
    )


# Preprocess data
def preprocess(dir_path):
    min_size = 800
    max_size = 1333
    image_mean = [0.485, 0.456, 0.406]  # ImageNet mean
    image_std = [0.229, 0.224, 0.225]  # ImageNet std
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    # Convert input images into a Pytorch Tensor
    fnames, original_imgs, scaled_images = load_image(dir_path)

    img_transformed, _ = transform(scaled_images, targets=None)
    return img_transformed


device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.detection.retinanet_resnet50_fpn(
    pretrained=False, num_classes=91, pretrained_backbone=False
)

dummy_input = torch.randn(1, 3, 800, 1333)

state_dict = torch.load(
    "ml/models/retinanet_resnet50_fpn_coco-eeacb38b.pth", map_location=device
)
model.load_state_dict(state_dict)
model.to(device).eval()
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {2: "height", 3: "width"},  # Dynamic height and width
        "output": {2: "height_out", 3: "width_out"},  # Adjust if output is dynamic
    },
    opset_version=11,  # Use opset supporting dynamic Resize (11+)
)
