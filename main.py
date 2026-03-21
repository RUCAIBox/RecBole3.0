import argparse
from recbole.utils import parse_command_line_args
from recbole.pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="HSTU", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Amazon2023", help="name of datasets"
    )
    parser.add_argument(
        "--category", "-c", type=str, default="Video_Games", help="category of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")

    args, unparsed_args = parser.parse_known_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        config_file_list=config_file_list,
        config_dict=command_line_configs
    )

    pipeline.run()

