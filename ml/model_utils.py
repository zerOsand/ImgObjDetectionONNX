import csv
import os

from .model_inputs import DetectionInputs
from .model_params import DetectionParameters


class Handler:
    pass


class Input_Handler(Handler):
    @staticmethod
    def parse_inputs(inputs: DetectionInputs) -> tuple[str, str, str, str]:
        input_path = inputs["input_path"].path
        output_img_path = inputs["output_img"].path
        output_csv_path = inputs["output_csv"].path
        model_path = inputs["model_path"].path

        return (input_path, output_img_path, output_csv_path, model_path)


class Parameter_Handler(Handler):
    @staticmethod
    def parse_parameters(parameters: DetectionParameters) -> int:
        min_perc_prob = parameters["min_perc_prob"]

        return min_perc_prob


class Results_Handler(Handler):
    @staticmethod
    def write_csv_results(results, output_csv_path) -> None:
        classes = [line.strip() for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco91_classes.txt")).readlines()]

        with open(output_csv_path, "w", newline="") as out:
            csv_writer = csv.writer(out)
            csv_writer.writerow(
                ["name", "percentage_probability", "xmin", "ymin", "xmax", "ymax"]
            )
            for item in results:
                csv_writer.writerow(
                    [
                        classes[item[2]],  # label
                        item[1],  # probability
                        item[0][0],  # xmin
                        item[0][1],  # ymin
                        item[0][2],  # xmax
                        item[0][3],  # ymax
                    ]
                )
