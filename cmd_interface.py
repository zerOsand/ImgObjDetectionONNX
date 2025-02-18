# Command line interface for image object detection
import argparse

from flask_ml.flask_ml_cli import MLCli

from backend.model_server import model_server


def main():
    parser = argparse.ArgumentParser(description="Command line interface for image object detection")
    cli = MLCli(model_server.server, parser)
    cli.run_cli()


if __name__ == "__main__":
    main()
