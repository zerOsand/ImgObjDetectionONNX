import os
import warnings

from imageai.Detection import ObjectDetection

from ml.model_utils import *

warnings.filterwarnings("ignore")

class Detection_Model:
    def __init__(self, model_type: Model_Type, model_path):
        self.detector_model = ObjectDetection()
        self.exec_path = os.getcwd()
        self.model_type = model_type
        self.model_path = model_path
        self.initialize()

    def initialize(self):
        Model_Handler.set_model_type(self.detector_model, self.model_type)

        full_model_path = os.path.abspath(os.path.join(self.exec_path, "ml", self.model_path))
        self.detector_model.setModelPath(full_model_path)
        self.detector_model.loadModel()

    def predict(self, input_image_path: str, output_image_path: str, min_perc_prob: int):
        return self.detector_model.detectObjectsFromImage(
            input_image=os.path.join(self.exec_path, input_image_path),
            output_image_path=os.path.join(self.exec_path, output_image_path),
            minimum_percentage_probability=min_perc_prob,
        )
