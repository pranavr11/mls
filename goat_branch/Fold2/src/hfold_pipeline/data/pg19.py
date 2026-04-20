from __future__ import annotations

import logging
from itertools import chain
from pathlib import Path

from ..config import DataConfig

logger = logging.getLogger(__name__)


def _limit_split(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_or_prepare_pg19(tokenizer, data_config: DataConfig):
    from datasets import DatasetDict, load_dataset, load_from_disk

    processed_path = Path(data_config.processed_dataset_dir)
    if processed_path.exists() and not data_config.overwrite_cache:
        logger.info("Loading processed dataset from %s", processed_path)
        dataset_dict = load_from_disk(str(processed_path))
        return _apply_split_limits(dataset_dict, data_config)

    logger.info(
        "Loading raw dataset %s (config=%s)",
        data_config.dataset_name,
        data_config.dataset_config_name,
    )
    raw_datasets = load_dataset(
        data_config.dataset_name,
        data_config.dataset_config_name,
    )

    split_names = list(raw_datasets.keys())
    example_split = raw_datasets[split_names[0]]
    remove_columns = example_split.column_names

    def tokenize_function(examples):
        return tokenizer(
            examples[data_config.text_column],
            return_attention_mask=False,
            truncation=False,
        )

    logger.info("Tokenizing PG-19 with %d workers", data_config.preprocessing_num_workers)
    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
        num_proc=data_config.preprocessing_num_workers,
        load_from_cache_file=not data_config.overwrite_cache,
        desc="Tokenizing PG-19",
    )

    block_size = data_config.block_size

    def group_texts(examples):
        concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["attention_mask"] = [
            [1] * block_size for _ in range(len(result["input_ids"]))
        ]
        result["labels"] = [chunk[:] for chunk in result["input_ids"]]
        return result

    logger.info("Chunking tokenized text into block_size=%d", block_size)
    lm_datasets = tokenized.map(
        group_texts,
        batched=True,
        num_proc=data_config.preprocessing_num_workers,
        load_from_cache_file=not data_config.overwrite_cache,
        desc=f"Grouping texts into chunks of {block_size}",
    )

    logger.info("Saving processed dataset to %s", processed_path)
    processed_path.mkdir(parents=True, exist_ok=True)
    lm_datasets.save_to_disk(str(processed_path))

    return _apply_split_limits(lm_datasets, data_config)


def _apply_split_limits(dataset_dict, data_config: DataConfig):
    from datasets import DatasetDict

    limited = DatasetDict()
    if data_config.train_split in dataset_dict:
        limited[data_config.train_split] = _limit_split(
            dataset_dict[data_config.train_split],
            data_config.max_train_samples,
        )
    if data_config.validation_split in dataset_dict:
        limited[data_config.validation_split] = _limit_split(
            dataset_dict[data_config.validation_split],
            data_config.max_validation_samples,
        )
    if data_config.test_split in dataset_dict:
        limited[data_config.test_split] = _limit_split(
            dataset_dict[data_config.test_split],
            data_config.max_test_samples,
        )
    return limited
