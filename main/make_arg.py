import argparse
import dataclasses
import itertools
import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from typing import Any, Dict, List, Tuple

from src.args import Config, get_config


def str2bool(val: str) -> bool:
    if isinstance(val, bool):
        return val
    if val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line interface return a list of parsed arguments."""

    parser = argparse.ArgumentParser(description="make configuration yaml files.")
    parser.add_argument('arg_file', type=str, help='choose arg_file([origin.yaml | tcn.yaml | new_origin.yaml | new.yaml])')
    parser.add_argument("--root_dir", type=str, default="./results", help="path to args file")

    fields = dataclasses.fields(Config)
    for field in fields:
        type_func = str2bool if field.type is bool else field.type
        parser.add_argument(f"--{field.name}", type=type_func, nargs="*", default=None)

    return parser.parse_args()


def parse_params(args_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[List[Any]]]:

    base_config = {}
    variable_keys = []
    variable_values = []

    for k, v in args_dict.items():
        if isinstance(v, list):
            variable_keys.append(k)
            variable_values.append(v)
        else:
            base_config[k] = v

    return base_config, variable_keys, variable_values


def main() -> None:
    # get args as dictionary.
    args = get_arguments()
    args_dict = vars(args).copy()
    del args_dict["arg_file"], args_dict["root_dir"]

    # load all args to args_dict
    arguements = get_config(os.path.join(args.root_dir, args.arg_file, "arg.yaml")) 
    for arg in args_dict:
        if args_dict[arg] is None:
            args_dict[arg] = dataclasses.asdict(arguements)[arg]

    base_config, variable_keys, variable_values = parse_params(args_dict)

    # get direct product
    product = itertools.product(*variable_values)

    # make a directory and save configuration file there.
    for values in product:
        config = base_config.copy()
        param_list = []
        for k, v in zip(variable_keys, values):
            config[k] = v
            param_list.append(f"{k}-{v}")

        dir_name = "_".join(param_list)
        dir_path = os.path.join(args.root_dir, args.arg_file + "+" +dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        config_path = os.path.join(dir_path, "arg.yaml")

        # save configuration file as yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("Finished making configuration files.")


if __name__ == "__main__":
    main()