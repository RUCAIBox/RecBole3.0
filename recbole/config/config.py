import re
import os
import sys
import yaml
from logging import getLogger
from typing import Literal
from recbole.utils import set_color, get_local_time


class Config(object):
    def __init__(
        self, model=None, dataset=None, config_file_list=None, config_dict=None
    ):
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.variable_config_dict = self._load_variable_config_dict(config_dict)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()

        self.model, self.dataset = self._get_model_and_dataset(
            model, dataset
        )
        self._load_internal_config_dict(self.model, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self.final_config_dict.update(config_dict)
        self.final_config_dict['run_local_time'] = get_local_time()
        self._init_device()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _load_variable_config_dict(self, config_dict):
        # HyperTuning may set the parameters such as mlp_hidden_size in NeuMF in the format of ['[]', '[]']
        # then config_dict will receive a str '[]', but indeed it's a list []
        # temporarily use _convert_config_dict to solve this problem
        return self._convert_config_dict(config_dict) if config_dict else dict()
    
    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type."""
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(
                    value, (str, int, float, list, tuple, dict, bool)
                ):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, "r", encoding="utf-8") as f:
                    file_config_dict.update(
                        yaml.load(f.read(), Loader=self.yaml_loader)
                    )
        return file_config_dict

    def _load_cmd_line(self):
        r"""Read parameters from command line and convert it to str."""
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if (
                    cmd_arg_name in cmd_config_dict
                    and cmd_arg_value != cmd_config_dict[cmd_arg_name]
                ):
                    raise SyntaxError(
                        "There are duplicate commend arg '%s' with different value."
                        % arg
                    )
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning(
                "command line args [{}] will not be used in RecBole".format(
                    " ".join(unrecognized_args)
                )
            )
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                final_model = self.external_config_dict["model"]
            except KeyError:
                raise KeyError(
                    "model need to be specified in at least one of the these ways: "
                    "[model variable, config file, config dict, command line] "
                )
        else:
            final_model = model

        if dataset is None:
            try:
                final_dataset = self.external_config_dict["dataset"]
            except KeyError:
                raise KeyError(
                    "dataset need to be specified in at least one of the these ways: "
                    "[dataset variable, config file, config dict, command line] "
                )
        else:
            final_dataset = dataset

        return final_model, final_dataset

    def _update_internal_config_dict(self, file):
        with open(file, "r", encoding="utf-8") as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            if config_dict is not None:
                self.internal_config_dict.update(config_dict)
        return config_dict

    def _load_internal_config_dict(self, model, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, "../props/default.yaml")
        model_init_file = os.path.join(
            current_path, "../props/model/" + model + ".yaml"
        )
        dataset_init_file = os.path.join(
            current_path, "../props/dataset/" + dataset + ".yaml"
        )

        self.internal_config_dict = dict()
        self.internal_config_dict["model"] = model
        self.internal_config_dict["dataset"] = dataset

        for file in [
            overall_init_file,
            model_init_file,
            dataset_init_file,
        ]:
            if os.path.isfile(file):
                config_dict = self._update_internal_config_dict(file)

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def _init_device(self):
        if isinstance(self.final_config_dict["gpu_id"], tuple):
            self.final_config_dict["gpu_id"] = ",".join(
                map(str, list(self.final_config_dict["gpu_id"]))
            )
        else:
            self.final_config_dict["gpu_id"] = str(self.final_config_dict["gpu_id"])
        gpu_id = self.final_config_dict["gpu_id"]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        self.final_config_dict["local_rank"] = 0
        self.final_config_dict["device"] = (
            torch.device("cpu")
            if len(gpu_id) == 0 or not torch.cuda.is_available()
            else torch.device("cuda")
        )

    def items(self):
        return self.final_config_dict.items()
    
    def get(self, key, default=None):
        return self.final_config_dict.get(key, default)
    
    def copy(self):
        return self.final_config_dict.copy()
    
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getattr__(self, item):
        if "final_config_dict" not in self.__dict__:
            raise AttributeError(
                f"'Config' object has no attribute 'final_config_dict'"
            )
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.final_config_dict.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = "\n"
        args_info += set_color("Hyper Parameters:\n", "pink")
        args_info += "\n".join(
            [
                (
                    set_color("{}", "cyan") + " =" + set_color(" {}", "yellow")
                ).format(arg, value)
                for arg, value in self.final_config_dict.items()
            ]
        )
        args_info += "\n\n"

        return args_info

    def __repr__(self):
        return self.__str__()


