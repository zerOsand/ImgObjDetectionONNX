from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    FileResponse,
    FileType,
    InputSchema,
    InputType,
    IntRangeDescriptor,
    ParameterSchema,
    RangedIntParameterDescriptor,
    ResponseBody,
    TaskSchema,
    NewFileInputType,
)

from ml.model_inputs import DetectionInputs
from ml.model_params import DetectionParameters
from ml.model_utils import *

from run_ONNX_model import ONNXDetectionModel


class Model_Server:

    server = MLServer(__name__)

    server.add_app_metadata(
        name="Image Object Detection",
        author="Shreneken",
        version="0.1.0",
        info="no",
    )

    @staticmethod
    def create_object_detection_task_schema() -> TaskSchema:
        input_path_schema = InputSchema(
            key="input_path", label="Input Image", input_type=InputType.FILE
        )
        output_img_schema = InputSchema(
            key="output_img",
            label="Output Image",
            input_type=NewFileInputType(
                allowed_extensions=[".png"], default_extension=".png"
            ),
        )
        output_csv_schema = InputSchema(
            key="output_csv",
            label="Output CSV",
            input_type=NewFileInputType(
                allowed_extensions=[".csv"], default_extension=".csv"
            ),
        )
        model_path_schema = InputSchema(
            key="model_path", label="ONNX Model", input_type=InputType.FILE
        )

        min_perc_prob_schema = ParameterSchema(
            key="min_perc_prob",
            label="Minimum Percentage Probability",
            value=RangedIntParameterDescriptor(
                range=IntRangeDescriptor(min=0, max=100), default=30
            ),
        )

        return TaskSchema(
            inputs=[
                input_path_schema,
                output_img_schema,
                output_csv_schema,
                model_path_schema,
            ],
            parameters=[min_perc_prob_schema],
        )

    @staticmethod
    @server.route(
        "/detect",
        task_schema_func=create_object_detection_task_schema,
        short_title="Detect Objects",
        order=0,
    )
    def detect(inputs: DetectionInputs, parameters: DetectionParameters):
        (
            input_path,
            output_img_path,
            output_csv_path,
            model_path,
        ) = Input_Handler.parse_inputs(inputs)
        min_perc_prob = Parameter_Handler.parse_parameters(parameters)

        model = ONNXDetectionModel(model_path)

        results = model.predict(input_path, min_perc_prob)

        model.visualize_and_save(input_path, results, output_img_path)

        Results_Handler.write_csv_results(results, output_csv_path)

        return ResponseBody(
            root=FileResponse(file_type=FileType.CSV, path=output_csv_path)
        )

    @classmethod
    def start_server(cls):
        cls.server.run()


model_server = Model_Server()

if __name__ == "__main__":
    model_server.start_server()
