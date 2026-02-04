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


def report_question_duplicates(split_records: dict[str, list[dict]]) -> None:
    print("\nDuplicate questions by split (index pairs):")
    any_found = False
    for split, records in split_records.items():
        question_to_indices: dict[str, list[int]] = {}
        for record in records:
            question = _to_str(record.get("question"))
            question_to_indices.setdefault(question, []).append(record["index"])

        dup_pairs = []
        for indices in question_to_indices.values():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        dup_pairs.append((indices[i], indices[j]))

        if dup_pairs:
            any_found = True
            print(f"  {split}: {len(dup_pairs)} pair(s)")
            for a, b in dup_pairs:
                print(f"    ({a}, {b})")
        else:
            print(f"  {split}: none")

    if not any_found:
        print("  (none found across all splits)")


def drop_and_reindex(
    split_records: dict[str, list[dict]],
    drop_indices_by_split: dict[str, set[int]],
) -> None:
    for split, records in split_records.items():
        drop_indices = drop_indices_by_split.get(split, set())
        filtered = [record for record in records if record["index"] not in drop_indices]
        for new_index, record in enumerate(filtered):
            record["source_index"] = record["index"]
            record["index"] = new_index
        split_records[split] = filtered


if __name__ == "__main__":
    missing = []
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        missing_str = "\n  - ".join(missing)
        raise SystemExit(f"Missing data files:\n  - {missing_str}\nRun ./run.sh to download them first.")

    split_records = {}
    for split, filename in SPLIT_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        records = load_jsonl_with_index(path)
        split_records[split] = records

    report_question_duplicates(split_records)

    drop_and_reindex(
        split_records,
        {
            "train": {7255, 8021},
            "us_qbank": {1176, 2197},
        },
    )

    print("\nDataset sizes (post-filter):")
    for split, records in split_records.items():
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
        desired_order = ["index", "source_index", "question", "options", "answer", "meta_info"]
        column_order = desired_order + [c for c in dataset.column_names if c not in desired_order]
        dataset = dataset.select_columns(column_order)
        print(f"Pushing '{split}' split...")
        dataset.push_to_hub(HF_REPO_ID, split=split, private=PRIVATE)

    print(f"Dataset pushed to https://huggingface.co/datasets/{HF_REPO_ID}")
