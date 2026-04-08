from logging import getLogger
import os
import json
from recbole.data.parser.amazon2023 import Amazon2023Parser
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
import copy
import torch


class SRDataset(TorchDataset):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()
        self.MAX_ITEM_SEQ_LEN = config['max_item_seq_len'] 

        self.all_item_seq = None
        self.id_mapping = None
        self.item2meta = None
        self.data = None
        
        # Check and load data using Parser if necessary
        self._check_and_load_data()

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id) - 1

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id) - 1

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for piece in self.all_item_seq:
            n_inters += len(piece['item_seq'])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping['item2id']

    def _get_parser(self):
        """
        Returns the appropriate parser based on the configuration.
        """
        if self.config['dataset'] == 'Amazon2023':
            return Amazon2023Parser(self.config)
        return None

    def _check_and_load_data(self):
        """
        Checks if the processed data exists. If not, runs the parser.
        Then loads the data.
        """
        parser = self._get_parser()
        if parser:
            output_path = parser.output_path
            seq_file = os.path.join(output_path, 'all_item_seq.json')
            
            if not os.path.exists(seq_file):
                self.log(f'[DATASET] Processed data not found at {output_path}. Starting parser...')
                parser.parse()
            else:
                self.log(f'[DATASET] Found processed data at {output_path}.')
            
            self._load_from_disk(output_path)
        else:
             self.log(f'[DATASET] No specific parser found for dataset: {self.config.get("dataset")}. Assuming data is managed elsewhere.')

    def _load_from_disk(self, output_path):
        """
        Loads the unified format data from the output path.
        """
        seq_file = os.path.join(output_path, 'all_item_seq.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        meta_file = os.path.join(output_path, f"metadata.{self.config.get('metadata', 'none')}.json")

        self.log(f'[DATASET] Loading data from {output_path}...')
        self.all_item_seq = json.load(open(seq_file, 'r'))
        self.id_mapping = json.load(open(id_mapping_file, 'r'))
        
        if os.path.exists(meta_file):
             self.item2meta = json.load(open(meta_file, 'r'))
        
    def _leave_one_out(self):
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        datasets = {'train': {'user': [], 'item_seq': [], 'timestamp_seq': []},
                    'val': {'user': [], 'item_seq': [], 'timestamp_seq': []},
                    'test': {'user': [], 'item_seq': [], 'timestamp_seq': []}}
        for piece in self.all_item_seq:
            user = piece['user_id']
            raw_item_seq = piece['item_seq']
            item_seq = list(map(lambda x: self.item2id[x], raw_item_seq))
            tsp_seq = list(map(lambda x: int(x), piece['timestamp_seq']))
            if len(item_seq) > 1:
                datasets['test']['user'].append(user)
                datasets['test']['item_seq'].append(item_seq)
                datasets['test']['timestamp_seq'].append(tsp_seq)
                if len(item_seq) > 2:
                    datasets['val']['user'].append(user)
                    datasets['val']['item_seq'].append(item_seq[:-1])
                    datasets['val']['timestamp_seq'].append(tsp_seq[:-1])
                if len(item_seq) > 3:
                    train_item_seq = item_seq[:-2]
                    train_tsp_seq = tsp_seq[:-2]
                    for i in range(1, len(train_item_seq)):
                        datasets['train']['user'].append(user)
                        datasets['train']['item_seq'].append(train_item_seq[:i+1])
                        datasets['train']['timestamp_seq'].append(train_tsp_seq[:i+1])
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets
    
    def _time_split(self):
        
        val_t, test_t = self.config['split_timestmap'][0], self.config['split_timestmap'][1]

        assert val_t < test_t and val_t > 0 and test_t > 0, \
            f"val_timestmap [{val_t}] must be less than test_timestmap [{test_t}]."

        datasets = {'train': {'user': [], 'item_seq': [], 'timestamp_seq': []},
                    'val': {'user': [], 'item_seq': [], 'timestamp_seq': []},
                    'test': {'user': [], 'item_seq': [], 'timestamp_seq': []}}

        for piece in self.all_item_seq:
            user = piece['user_id']
            raw_item_seq = piece['item_seq']
            item_seq = list(map(lambda x: self.item2id[x], raw_item_seq))
            tsp_seq = list(map(lambda x: int(x), piece['timestamp_seq']))
            slen = len(item_seq)

            val_start = sum([t < val_t for t in tsp_seq])
            test_start = sum([t < test_t for t in tsp_seq])
            for i in range(1, val_start):
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(item_seq[:i+1])
                datasets['train']['timestamp_seq'].append(tsp_seq[:i+1])
            for i in range(val_start, test_start):
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(item_seq[:i+1])
                datasets['val']['timestamp_seq'].append(tsp_seq[:i+1])
            for i in range(test_start, slen):
                datasets['test']['user'].append(user)
                datasets['test']['item_seq'].append(item_seq[:i+1])
                datasets['test']['timestamp_seq'].append(tsp_seq[:i+1])
        
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def __len__(self):
        return len(self.data)
    
    def split(self):
        """
        Splits the dataset into train, validation, and test sets using the specified split strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """

        split_strategy = self.config['split']
        if split_strategy == 'leave_one_out':
            datasets = self._leave_one_out()
        elif split_strategy == 'time_split':
            datasets = self._time_split()
        else:
            # If split logic was handled by parser (e.g. timestamp split) and saved/loaded, we might need to handle it.
            # But currently we only loaded all_item_seqs.
            raise NotImplementedError(f'Split strategy [{split_strategy}] not implemented.')

        split_data = [self.copy(_) for _ in datasets.values()]
        return split_data

    def __getitem__(self, index):
        piece = self.data[index]
        all_item_seq = piece['item_seq']
        all_tsp_seq = piece['timestamp_seq']
        item_seq = all_item_seq[:-1][-self.MAX_ITEM_SEQ_LEN:]
        item = all_item_seq[-1:]
        tsp_seq = all_tsp_seq[:-1][-self.MAX_ITEM_SEQ_LEN:]
        tsp = all_tsp_seq[-1:]

        return item_seq, item, tsp_seq, tsp
    
    def copy(self, data):
        nxt = copy.copy(self)
        nxt.data = data
        return nxt
    
    def log(self, message, level='info'):
        try:
            from recbole.utils import log
            return log(message, self.config['accelerator'], self.logger, level=level)
        except ImportError:
            getattr(self.logger, level)(message)


if __name__ == '__main__':
    # Test code for SRDataset and Amazon2023Parser
    import os
    from recbole.config.config import Config

    def test_amazon2023_parser_and_dataset():
        """
        Test the Amazon2023Parser and SRDataset functionality.
        """
        print("=" * 60)
        print("Testing SRDataset with Amazon2023Parser")
        print("=" * 60)


        try:
            # Create config for testing
            config_dict = {
                'model': 'HSTU',
                'dataset': 'Amazon2023',
                'category': 'Musical_Instruments',  # Small category for quick testing
                'kcore': '5core',
                'cache_dir': './cache',
                'metadata': 'none',
                # 'split': 'leave_one_out',
                'split': 'time_split',
                'split_timestmap': [1567619784438, 1600784937055],
                'accelerator': None,
                'num_proc': 1,
                'gpu_id': 0,
            }

            config = Config(config_dict=config_dict)
            print(f"\n[TEST] Config created:")
            print(f"  - Dataset: {config['dataset']}")
            print(f"  - Category: {config['category']}")
            print(f"  - K-core: {config['kcore']}")
            print(f"  - Cache dir: {config['cache_dir']}")

            # Initialize dataset (this will trigger Amazon2023Parser)
            print("\n[TEST] Initializing SRDataset (will trigger parser if data not found)...")
            dataset = SRDataset(config)

            # Print dataset info
            print(f"\n{dataset}")

            # Test data splitting
            print("\n[TEST] Splitting dataset using leave_one_out strategy...")
            split_data = dataset.split()

            print(f"\n[TEST] Split results:")
            for split_dataset in split_data:
                print(f"  - : {len(split_dataset)} samples")

            print("\n[TEST] First five samples in train dataset:")
            for i in range(5):
                print(split_data[0][i])

            print("\n" + "=" * 60)
            print("[TEST] All tests passed successfully!")
            print("=" * 60)

        except Exception as e:
            print(f"\n[TEST] Error occurred: {e}")
            import traceback
            traceback.print_exc()

    # Run the test
    test_amazon2023_parser_and_dataset()
