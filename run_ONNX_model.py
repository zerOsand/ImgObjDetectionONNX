import os
import cv2
import uuid
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import Union, List, Tuple
import torchvision.transforms.v2 as T
from torchvision.io import ImageReadMode, image


class ObjectDetectionProcessing:
    def __init__(self, resolution=800):
        self.resolution = resolution
        self.valid_extensions = ("jpg", "jpeg", "png")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Variables to store preprocessing parameters
        self.original_height = None
        self.original_width = None
        self.scale = None
        self.x_offset = None
        self.y_offset = None
        self.orig_img = None

    def postprocess_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Postprocess bounding boxes to original image dimensions."""
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        boxes = boxes.astype(np.float32)
        # Scale back to original dimensions
        boxes /= self.scale
        # Clip to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.original_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.original_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.original_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.original_height)
        return boxes.astype(np.int32)

    def draw_caption(self, image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(
            image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2
        )
        cv2.putText(
            image,
            caption,
            (b[0], b[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
        )

    def preprocess(self, path: str) -> np.ndarray:
        """Preprocess image to 800x1333 - dependent on small/largest sides"""
        if os.path.isfile(path):
            if path.rsplit(".")[-1].lower() not in self.valid_extensions:
                raise ValueError("Unsupported image type")

        image = cv2.imread(path)

        self.orig_img = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 800
        max_side = 1333
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        self.scale = scale
        # resize the image with the computed scale
        image = cv2.resize(
            image, (int(round(cols * scale)), int(round((rows * scale))))
        )
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        return image

    def save_temp_img(self, input_image: Union[np.ndarray, Image.Image]) -> str:
        """Save in-memory image to a temporary file."""
        temp_path = os.path.join(os.path.dirname(__file__), f"{uuid.uuid4()}.jpg")
        if isinstance(input_image, np.ndarray):
            cv2.imwrite(temp_path, input_image)
        elif isinstance(input_image, Image.Image):
            input_image.save(temp_path)
        else:
            raise ValueError("Unsupported image type")
        return temp_path

    def read_image(
        self, path: str, mode: ImageReadMode = ImageReadMode.UNCHANGED
    ) -> torch.Tensor:
        """Read image into a tensor."""
        data = image.read_file(path)
        return image.decode_image(data, mode)

    def apply_transforms(self, image: torch.Tensor) -> torch.Tensor:
        """Resize and crop the image to 800x800."""
        transform = T.Compose(
            [
                T.Resize(
                    self.resolution + self.resolution // 8,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
            ]
        )
        return transform(image)


class ONNXDetectionModel:
    def __init__(self, model_path: str):
        self.odp = ObjectDetectionProcessing()

        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.__classes = self.__load_classes(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "coco91_classes.txt"
            )
        )

    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes

    def predict(
        self, image_path: str, min_confidence: float
    ) -> List[Tuple[list, float, int]]:
        """Run object detection and return scaled bounding boxes."""
        input_tensor = self.odp.preprocess(image_path)
        outputs = self.session.run(None, {"input": input_tensor})
        if not outputs:
            return []
        boxes, scores, labels = outputs[0], outputs[1], outputs[2]
        # Filter based on confidence
        mask = scores >= (min_confidence / 100)
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        # Postprocess boxes
        scaled_boxes = self.odp.postprocess_boxes(filtered_boxes)
        # Format results
        results = []
        for i in range(len(filtered_scores)):
            box = scaled_boxes[i].tolist()
            score = filtered_scores[i].item()
            label = filtered_labels[i].item()
            results.append((box, score, label))
        return results

    def visualize_and_save(
        self,
        image_path: str,
        detections: List[Tuple[list, float, int]],
        output_path: str,
    ):
        """Overlay bounding boxes on the original image and save it."""
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Draw bounding boxes and labels
        for box, score, label in detections:
            x1, y1, x2, y2 = box
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label and confidence score
            label_text = f"{self.__classes[label]}: {score:.2f}"
            print(f"{score:.2f}")
            cv2.putText(
                image,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        # Save the image with overlaid boxes
        cv2.imwrite(output_path, image)
        print(f"Saved visualization to {output_path}")


# # Example usage
# image_path = 'input_images/carvision.png'

# output_path = 'test_out/carvision_w_boxes.png'

# threshold = 20 # as a percent

# model = ONNXDetectionModel('model.onnx')

# detections = model.predict(image_path, threshold)

# import pdb; pdb.set_trace()

# model.visualize_and_save(image_path, detections, output_path)
