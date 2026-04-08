import os
import json
import logging
from recbole.data.parser.base import BaseParser
from datasets import load_dataset
from recbole.utils.utils import clean_text


class Amazon2023Parser(BaseParser):
    """
    Parser for the Amazon 2023 dataset.
    Converts raw complex JSONL files from Amazon 2023 into the unified standard format.
    """
    def __init__(self, config):
        self.config = config
        self.category = config['category']
        self.logger = logging.getLogger()
        
        # Determine paths
        # Using config['cache_dir'] as base for raw downloads
        self.cache_dir = os.path.join(
            config['cache_dir'], 'AmazonReviews2023', self.category
        )
        
        # Output path is where we save the processed unified files
        # We can use the same directory structure or a specific output path
        # Here we follow the logic: cache_dir/processed
        output_path = os.path.join(self.cache_dir, 'processed')
        
        # Raw data path is implicit in load_dataset cache or specific dir
        raw_data_path = os.path.join(
            self.cache_dir,
            f"raw/benchmark/{self.config['kcore']}/full_rating_only"
        )
        
        super().__init__(raw_data_path, output_path)
        
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.all_item_seq = []

    def parse(self):
        """
        Downloads and processes the raw data for the AmazonReviews2023 dataset.
        Saves the processed data (id_mapping, metadata, all_item_seqs) to self.output_path.
        """
        self._check_available_category()
        
        # Download/Load Raw Data
        if os.path.isdir(self.raw_data_path):
            dataset = load_dataset('csv', data_file=os.path.join(self.raw_data_path, f'{self.category}.csv'))
            self.logger.info(f'[DATASET] Loading raw data from {self.raw_data_path}')
        else:
            # Download processed reviews
            # Assuming accelerator is available in config if needed, or just standard download
            # If distributed, care must be taken. For parser, maybe we assume single process or lock.
            accelerator = self.config['accelerator']
            if accelerator:
                with accelerator.main_process_first():
                    dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        f"{self.config['kcore']}_rating_only_{self.category}",
                        split='full',
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )
            else:
                dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    f"{self.config['kcore']}_rating_only_{self.category}",
                    split='full',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )

        # Process & Save ID Mapping
        self.id_mapping = self._remap_ids(dataset, self.output_path)

        # Process & Save Metadata
        self.item2meta = self._process_meta(self.output_path)

        self.all_item_seq = self._process_all_item_seq(dataset, self.output_path)
        
        self.logger.info(f"[PARSER] Amazon 2023 dataset parsing complete! Unified data is ready at '{self.output_path}'")

    def _check_available_category(self):
        available_categories = [
            'All_Beauty', 'Amazon_Fashion', 'Appliances', 'Art_Crafts_and_Sewing',
            'Automotive', 'Baby_Products', 'Beauty_and_Personal_Care', 'Books',
            'CDs_and_Vinyl', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry',
            'Digital_Music', 'Electronics', 'Gift_Cards', 'Grocery_and_Gourmet_Food',
            'Handmade_Products', 'Health_and_Household', 'Health_and_Personal_Care',
            'Home_and_Kitchen', 'Industrial_and_Scientific', 'Kindle_Store',
            'Magazine_Subscriptions', 'Movies_and_TV', 'Musical_Instruments',
            'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Software',
            'Sports_and_Outdoors', 'Subscription_Boxes', 'Tools_and_Home_Improvement',
            'Toys_and_Games', 'Unknown', 'Video_Games',
        ]
        assert self.category in available_categories, \
            f'Category "{self.category}" not available. Available categories: {available_categories}'

        if self.config['kcore'] == '5core':
            if self.category in [
                'Amazon_Fashion', 'Appliances', 'Digital_Music', 'Handmade_Products',
                'Health_and_Personal_Care', 'Subscription_Boxes',
            ]:
                raise ValueError(
                    f'[PARSER] Category "{self.category}" does not have 5-core reviews.'
                )

    def _remap_ids(self, dataset, output_path: str):
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        # Logic: If file exists, we might want to skip, but parse() is usually called when we need to regenerate 
        # or we can trust the caller to check existence. 
        # However, to be safe and consistent with previous logic:
        if os.path.exists(id_mapping_file):
            self.logger.info(f'[PARSER] Loading id mapping from {id_mapping_file}')
            return json.load(open(id_mapping_file, 'r'))

        for user_id, item_id, rating, timestamp in zip(
            dataset['user_id'],
            dataset['parent_asin'],
            dataset['rating'],
            dataset['timestamp'],
        ):
            if user_id not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user_id] = len(self.id_mapping['user2id'])
                self.id_mapping['id2user'].append(user_id)
            if item_id not in self.id_mapping['item2id']:
                self.id_mapping['item2id'][item_id] = len(self.id_mapping['item2id'])
                self.id_mapping['id2item'].append(item_id)

        with open(id_mapping_file, 'w') as f:
            json.dump(self.id_mapping, f)
        return self.id_mapping

    def _feature_process(self, feature):
        sentence = ""
        if isinstance(feature, float):
            sentence += str(feature) + '.'
        elif isinstance(feature, list) and len(feature) > 0:
            for v in feature:
                sentence += clean_text(v) + ', '
            sentence = sentence[:-2] + '.'
        else:
            sentence = clean_text(feature)
        return sentence + ' '

    def _clean_metadata(self, example):
        meta_text = ''
        features_needed = ['title', 'features', 'categories', 'description']
        for feature in features_needed:
            meta_text += self._feature_process(example[feature])
        example['cleaned_metadata'] = meta_text
        return example

    def _extract_meta_sentences(self, meta_dataset):
        meta_dataset = meta_dataset.map(
            lambda t: self._clean_metadata(t),
            num_proc=self.config['num_proc']
        )
        item2meta = {}
        for parent_asin, cleaned_metadata in zip(
            meta_dataset['parent_asin'],
            meta_dataset['cleaned_metadata']
        ):
            item2meta[parent_asin] = cleaned_metadata
        return item2meta

    def _process_meta(self, output_path: str):
        process_mode = self.config['metadata']
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
        if os.path.exists(meta_file):
            self.logger.info(f'[PARSER] Metadata has been processed...')
            return json.load(open(meta_file, 'r'))

        self.logger.info(f'[PARSER] Processing metadata, mode: {process_mode}')

        if process_mode == 'none':
            return None

        meta_dataset = load_dataset(
            'McAuley-Lab/Amazon-Reviews-2023',
            f'raw_meta_{self.category}',
            split='full',
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        meta_dataset = meta_dataset.filter(
            lambda t: t['parent_asin'] in self.id_mapping['item2id']
        )
        self.logger.info(
            f'[PARSER] {len(meta_dataset)} of '
            f'{len(self.id_mapping["item2id"]) - 1} items have meta data.'
        )

        if process_mode == 'sentence':
            item2meta = self._extract_meta_sentences(meta_dataset=meta_dataset)
        else:
            raise NotImplementedError(f'Metadata processing mode "{process_mode}" not implemented.')

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        return item2meta

    def _process_all_item_seq(self, dataset, output_path: str):
        all_item_seq_file = os.path.join(output_path, 'all_item_seq.json')
        if os.path.exists(all_item_seq_file):
            self.logger.info(f'[PARSER] All item seq has been processed...')
            return json.load(open(all_item_seq_file, 'r'))
        
        df = dataset.to_pandas()
        df_sorted = df.sort_values(
            by=["user_id", "timestamp"],
            ascending=[True, True],
            ignore_index=True
        )
        
        # aggregate by user_id, sort by timestamp
        user_seq_df = df_sorted.groupby("user_id", as_index=False).agg(
            item_seq=("parent_asin", list),
            timestamp_seq=("timestamp", list)
        )
        
        # convert to dict list
        self.all_item_seq = user_seq_df.to_dict(orient="records")

        with open(all_item_seq_file, 'w') as f:
            json.dump(self.all_item_seq, f)
        
        return self.all_item_seq

        


