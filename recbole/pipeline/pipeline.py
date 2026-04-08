from logging import getLogger
import torch
import sys
import os
from accelerate import Accelerator
from recbole.config import Config
from recbole.utils import init_seed, init_logger, init_device, get_model, get_trainer, log
from recbole.data.utils import get_dataset, data_preparation


class Pipeline:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        config_dict: dict = None,
        config_file_list: str = None,
    ):
        self.config = Config(
            model=model_name,
            dataset=dataset_name,
            config_file_list=config_file_list,
            config_dict=config_dict
        )
        # Automatically set devices and ddp
        self.config['device'], self.config['use_ddp'] = init_device() 

        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )
        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator

        # Seed and Logger
        init_seed(self.config['seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(sys.argv)
        self.log(self.config)
        self.log(f'Device: {self.config["device"]}')

        # Dataset
        self.raw_dataset = get_dataset(self.config)(self.config)
        self.log(self.raw_dataset)
        
        # dataset splitting
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.raw_dataset)

        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(self.config["model"])(self.config, self.raw_dataset).to(self.config["device"])
        self.log(self.model)

        self.trainer = get_trainer(self.config["model"])(self.config, self.model)

    def run(self):

        self.trainer.fit(self.train_data, self.valid_data)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        self.model, self.test_data = self.accelerator.prepare(
            self.model, self.test_data
        )
        if self.accelerator.is_main_process:
            self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        test_results = self.trainer.evaluate(self.test_data)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        self.trainer.end()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
