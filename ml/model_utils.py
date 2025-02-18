import csv
from enum import Enum, auto

from .model_inputs import DetectionInputs
from .model_params import DetectionParameters


class Model_Path(Enum):
    yolov3 = "./models/yolov3.pt"
    tiny_yolov3 = "./models/tiny-yolov3.pt"
    retina_net = "./models/retinanet_resnet50_fpn_coco-eeacb38b.pth"


class Model_Type(Enum):
    yolov3 = auto()
    tiny_yolov3 = auto()
    retina_net = auto()


class Handler:
    pass


class Model_Handler(Handler):
    @staticmethod
    def set_model_type(model, model_type):
        if model_type == Model_Type.yolov3:
            model.setModelTypeAsYOLOv3()
        elif model_type == Model_Type.tiny_yolov3:
            model.setModelTypeAsTinyYOLOv3()
        elif model_type == Model_Type.retina_net:
            model.setModelTypeAsRetinaNet()
        else:
            raise Exception("No such model type!")

    @staticmethod
    def get_type(type_in_str: str):
        result = (
            Model_Type.yolov3
            if type_in_str == "yolov3"
            else (
                Model_Type.tiny_yolov3
                if type_in_str == "tiny-yolov3"
                else (
                    Model_Type.retina_net if type_in_str == "retina-net" else Exception("No such model type!")
                )
            )
        )
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def get_path(model_type):
        result = (
            Model_Path.retina_net
            if model_type == Model_Type.retina_net
            else (
                Model_Path.tiny_yolov3
                if model_type == Model_Type.tiny_yolov3
                else Model_Path.yolov3 if model_type == Model_Type.yolov3 else Exception("No such model!")
            )
        )
        if isinstance(result, Exception):
            raise result
        return result


class Input_Handler(Handler):
    @staticmethod
    def parse_inputs(inputs: DetectionInputs) -> tuple[str, str, str]:

        input_path = inputs["input_path"].path
        output_img_path = inputs["output_img"].path
        output_csv_path = inputs["output_csv"].path

        return (input_path, output_img_path, output_csv_path)


class Parameter_Handler(Handler):
    @staticmethod
    def parse_parameters(parameters: DetectionParameters) -> tuple[int, Model_Type, str]:
        min_perc_prob = parameters["min_perc_prob"]
        model_type = Model_Handler.get_type(parameters["model_type"])
        model_path = Model_Handler.get_path(model_type).value

        return (min_perc_prob, model_type, model_path)


class Results_Handler(Handler):
    @staticmethod
    def write_csv_results(results, output_csv_path) -> None:
        with open(output_csv_path, "w", newline="") as out:
            csv_writer = csv.writer(out)
            csv_writer.writerow(["name", "percentage_probability", "xmin", "ymin", "xmax", "ymax"])
            for item in results:
                csv_writer.writerow(
                    [
                        item["name"],
                        item["percentage_probability"],
                        item["box_points"][0],  # xmin
                        item["box_points"][1],  # ymin
                        item["box_points"][2],  # xmax
                        item["box_points"][3],  # ymax
                    ]
                )
