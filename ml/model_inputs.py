from typing import TypedDict

from flask_ml.flask_ml_server.models import FileInput


class DetectionInputs(TypedDict):
    input_path: FileInput
    output_img: FileInput
    output_csv: FileInput
