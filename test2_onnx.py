import os
import cv2
import uuid
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Tuple
import onnxruntime as ort
import torchvision.transforms.v2 as T
from torchvision.io import ImageReadMode, image
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import albumentations as A

import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

class ObjectDetectionProcessing:
    def __init__(self, resolution=800):
        self.resolution = resolution
        self.valid_extensions = ("jpg", "jpeg", "png")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Variables to store preprocessing parameters
        self.original_height = None
        self.original_width = None
        self.scale = None
        # self.x_offset = None
        # self.y_offset = None

        self.orig_img = None

    def postprocess_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Postprocess bounding boxes to original image dimensions."""
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        boxes = boxes.astype(np.float32)
        # Adjust for crop offset
        # boxes[:, [0, 2]] += self.x_offset  # x coordinates
        # boxes[:, [1, 3]] += self.y_offset  # y coordinates
        # Scale back to original dimensions
        boxes /= self.scale
        # Clip to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.original_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.original_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.original_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.original_height)
        return boxes.astype(np.int32)

    def _postprocess_boxes(self, filtered_boxes, filtered_scores, filtered_labels):
        outputs = {}
        outputs['pred_boxes'] = filtered_boxes
        outputs['pred_logits'] = filtered_scores
        outputs['pred_labels'] = filtered_labels
        
    def preprocess(self, path: str):
        if os.path.isfile(path):
            if path.rsplit('.')[-1].lower() in self.valid_extensions:
                self.orig_img = cv2.imread(path)
                frame_height, frame_width, _ = self.orig_img.shape

                RESIZE_TO = 800

                image_resized = self.resize(self.orig_img, RESIZE_TO, square=True)
                image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image = image / 255.0
                image = self.infer_transforms(image)
                input_tensor = torch.tensor(image, dtype=torch.float32)
                input_tensor = torch.permute(input_tensor, (2, 0, 1))
                input_tensor = input_tensor.unsqueeze(0)

                self.original_height = self.orig_img.shape[1]
                self.original_width = self.orig_img.shape[2]

                min_dim = min(self.original_height, self.original_width)
                self.scale = 800.0 / min_dim  # Resize shorter side to 800
                
                return input_tensor.detach().cpu().numpy() if input_tensor.requires_grad else input_tensor.cpu().numpy()
    

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def infer_transforms(self, image):
        transform = A.Compose([
            A.Normalize(max_pixel_value=1),
        ])
        return transform(image=image)['image']
        
    def resize(self, im, img_size=640, square=False):
        # Aspect ratio resize
        if square:
            im = cv2.resize(im, (img_size, img_size))
        else:
            h0, w0 = im.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
        return im

class ONNXDetectionModel:
    def __init__(self, model_path: str):
        self.odp = ObjectDetectionProcessing()

        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.__classes = self.__load_classes(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco91_classes.txt"))


    def __load_classes(self, path: str) -> List[str]:
        with open(path) as f:
            unique_classes = [c.strip() for c in f.readlines()]
        return unique_classes
    
    def predict(self, image_path: str, min_confidence: float) -> List[Tuple[list, float, int]]:
        """Run object detection and return scaled bounding boxes."""
        input_tensor = self.odp.preprocess(image_path)
        outputs = self.session.run(None, {"input": input_tensor})

        # import pdb; pdb.set_trace()

        if not outputs:
            return []
        boxes, scores, labels = outputs[0], outputs[1], outputs[2]
        # Filter based on confidence
        mask = scores >= min_confidence
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
            results.append( (box, score, label) )
        return results
    
    def visualize_and_save(self, image_path: str, detections: List[Tuple[list, float, int]], output_path: str):
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
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the image with overlaid boxes
        cv2.imwrite(output_path, image)
        print(f"Saved visualization to {output_path}")

# Example usage
image_path = 'input_images/cars.png'

output_path = 'test_out/cars_with_boxes.png'

model = ONNXDetectionModel('obj_detection_model.onnx')

detections = model.predict(image_path, 0.0)

model.visualize_and_save(image_path, detections, output_path)

#-----
