from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset


SCROLLS_TASKS = {"gov_report", "qasper"}


def _scrolls_text(example: Dict) -> str:
    source = example.get("input", "")
    output = example.get("output", "")
    if isinstance(output, list):
        output = "\n".join([str(x) for x in output])
    return f"{source}\n\n{output}".strip()


def load_raw_dataset(dataset_name: str, scrolls_task: str, cache_dir: str) -> DatasetDict:
    if dataset_name == "pg19":
        try:
            ds = load_dataset("pg19", cache_dir=cache_dir)
        except RuntimeError as e:
            # HF datasets>=4 dropped script-based loading; PG-19 still uses a script.
            if "Dataset scripts are no longer supported" in str(e):
                raise RuntimeError(
                    "PG-19 requires huggingface `datasets<4`. "
                    "Please run: pip install 'datasets>=2.18,<4'"
                ) from e
            raise
        if "validation" not in ds and "train" in ds:
            split = ds["train"].train_test_split(test_size=0.01, seed=42)
            ds = DatasetDict(train=split["train"], validation=split["test"])
        return ds

    if dataset_name == "scrolls":
        if scrolls_task not in SCROLLS_TASKS:
            raise ValueError(f"Unsupported SCROLLS task: {scrolls_task}")
        ds = load_dataset("tau/scrolls", scrolls_task, cache_dir=cache_dir)
        if "validation" not in ds:
            split = ds["train"].train_test_split(test_size=0.05, seed=42)
            ds = DatasetDict(train=split["train"], validation=split["test"])
        return ds

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _extract_texts(dataset_name: str, split: Dataset) -> List[str]:
    texts = []
    for row in split:
        if dataset_name == "pg19":
            t = row.get("text", "")
        else:
            t = _scrolls_text(row)
        if t and isinstance(t, str):
            texts.append(t)
    return texts


def tokenize_and_chunk(dataset_name: str, ds: DatasetDict, tokenizer, block_size: int):
    tokenized = {}
    for split_name in ["train", "validation"]:
        texts = _extract_texts(dataset_name, ds[split_name])
        all_ids = []
        for t in texts:
            all_ids.extend(tokenizer.encode(t, add_special_tokens=False))
            all_ids.append(tokenizer.eos_token_id)

        num_blocks = len(all_ids) // block_size
        usable = all_ids[: num_blocks * block_size]
        blocks = [usable[i : i + block_size] for i in range(0, len(usable), block_size)]
        tokenized[split_name] = Dataset.from_dict({"input_ids": blocks})
    return DatasetDict(tokenized)
