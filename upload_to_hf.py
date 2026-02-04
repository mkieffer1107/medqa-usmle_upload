#!/usr/bin/env python3
import os
import json
import argparse

from datasets import Dataset
from huggingface_hub import create_repo


def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


parser = argparse.ArgumentParser(description="Format MedQA-USMLE data and push to HF")
parser.add_argument("--hf_username", default=os.environ.get("HF_USERNAME", "mkieffer"))
parser.add_argument("--hf_repo_name", default=os.environ.get("HF_REPO_NAME", "MedQA-USMLE"))
parser.add_argument(
    "--private",
    default=os.environ.get("PRIVATE", "false"),
    help="Whether the HF dataset repo should be private (true/false)",
)
args = parser.parse_args()

HF_USERNAME = args.hf_username
HF_REPO_NAME = args.hf_repo_name
PRIVATE = str_to_bool(args.private)
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"

DATA_DIR = "data"
OUT_DIR = "output"
SPLIT_FILES = {
    "train": "train.jsonl",
    "test": "test.jsonl",
    "dev": "dev.jsonl",
    "us_qbank": "us_qbank.jsonl",
}


def _to_str(value) -> str:
    return "" if value is None else str(value)


def load_jsonl_with_index(file_path: str) -> list[dict]:
    records = []
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record = {"index": i}
            if isinstance(obj, dict):
                if "index" in obj:
                    obj = {k: v for k, v in obj.items() if k != "index"}
                # Normalize answer to be the choice letter for all splits.
                if "answer_idx" in obj and obj["answer_idx"] is not None:
                    obj["answer"] = obj["answer_idx"]
                    obj.pop("answer_idx", None)
                record.update(obj)
            else:
                record["value"] = obj
            records.append(record)
    return records


def save_jsonl(records: list[dict], file_path: str) -> None:
    with open(file_path, "w") as file:
        for record in records:
            json.dump(record, file)
            file.write("\n")


def normalize_options(records: list[dict], option_keys: list[str]) -> None:
    for record in records:
        options = record.get("options")
        if not isinstance(options, dict):
            options = {}
        normalized = {}
        for key in option_keys:
            normalized[key] = _to_str(options.get(key, ""))
        record["options"] = normalized


if __name__ == "__main__":
    missing = []
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        missing_str = "\n  - ".join(missing)
        raise SystemExit(f"Missing data files:\n  - {missing_str}\nRun ./run.sh to download them first.")

    print("Dataset sizes:")
    split_records = {}
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        records = load_jsonl_with_index(path)
        split_records[split] = records
        print(f"  {split}: {len(records)}")

    option_keys = set()
    for records in split_records.values():
        for record in records:
            opts = record.get("options")
            if isinstance(opts, dict):
                option_keys.update(opts.keys())

    option_keys = sorted(option_keys)
    for records in split_records.values():
        normalize_options(records, option_keys)

    os.makedirs(OUT_DIR, exist_ok=True)
    for split, filename in SPLIT_FILES.items():
        save_jsonl(split_records[split], os.path.join(OUT_DIR, filename))

    print(f"\nPushing dataset to Hugging Face Hub as {HF_REPO_ID} (private={PRIVATE})...")
    create_repo(HF_REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)

    for split, records in split_records.items():
        dataset = Dataset.from_list(records)
        desired_order = ["index", "question", "options", "answer", "meta_info"]
        column_order = desired_order + [c for c in dataset.column_names if c not in desired_order]
        dataset = dataset.select_columns(column_order)
        print(f"Pushing '{split}' split...")
        dataset.push_to_hub(HF_REPO_ID, split=split, private=PRIVATE)

    print(f"Dataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")
