import os
import cv2
import uuid
import torch
import numpy as np
from PIL import Image
from typing import Union
import onnxruntime as ort
import torchvision.transforms.v2 as T
from torchvision.io import ImageReadMode, image
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import AnchorGenerator

class ObjectDetectionProcessing:
    def __init__(self, resolution=800):
        self.resolution = resolution
        self.valid_extenions = (".jpg", ".jpeg", ".png")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # For postprocessing; gets defined in `preprocess` after image path is defined
        self.path = None

    def postprocess(self, output): 
        # todo
        return {}

    def preprocess(self, path):
        min_size = 800
        max_size = 1333
        image_mean = [0.485, 0.456, 0.406]  # ImageNet mean
        image_std = [0.229, 0.224, 0.225]   # ImageNet std
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        # Convert input images into a Pytorch Tensor
        fnames, original_imgs, scaled_images = self.load_image(path) 

        img_transformed, _ = transform(scaled_images, targets=None)

        self.path = path

        return img_transformed.tensors.numpy()
    
    def anchorgen(self):
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        return anchor_generator

    def load_image(self, input_image):
        """
        Loads image from the given path.
        """
        allowed_file_extensions = ("jpg", "jpeg", "png")
        images = []
        scaled_images = []
        fnames = []
        
        delete_file = False
        if type(input_image) is not str:
            input_image = self.save_temp_img(input_image=input_image)
            delete_file = True

        if os.path.isfile(input_image):
            if input_image.rsplit('.')[-1].lower() in allowed_file_extensions:
                img = self.read_image(input_image, ImageReadMode.RGB)
                img = self.apply_transforms(img)

                images.append(img)
                scaled_images.append(img.div(255.0).to(self.device))
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

    def save_temp_img(self, input_image : Union[np.ndarray, Image.Image]) -> str:

        temp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{str(uuid.uuid4())}.jpg" 
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
    
    def read_image(self, path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> torch.Tensor:
        data = image.read_file(path)
        return image.decode_image(data, mode)

    def apply_transforms(self, image):
        _transform = T.Compose(
            [
                T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
            ])
        return _transform(image)

class ONNXDetectionModel:
    def __init__(self, model_path):
        self.odp = ObjectDetectionProcessing()
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def predict(self, image_path: str, min_percent: float):
        input = self.odp.preprocess(image_path)
        output = self.session.run(None, {"input": input})

        if output is None:
            return []
    
        # Remove values <= min_percent
        mask = output[1] >= min_percent

        output = [output[0][mask],
                  output[1][mask],
                  output[2][mask]]

        # for idx, pred in enumerate(output):
        #     for id in range(pred["labels"].shape[0]):
        #         if pred["scores"][id] >= self.

        # out = self.odp.postprocess(output[0]) # todo
        # out["image_path"] = image_path
        # return out
        import pdb; pdb.set_trace()

    def predict_dir(self, input_dir):
        outputs = []
        # todo
        # for inp in self.odp.find_images_in_dir(input_dir):
        #   outputs.append(self.predict(inp))
        return outputs
    


model = ONNXDetectionModel('obj_detection_model.onnx')
model.predict('input_images/cars.png', 0.0)