from datasets import load_dataset

if __name__ == "__main__":
    splits = ["train", "dev", "test", "us_qbank"]

    train_ds = load_dataset("mkieffer/MedQA-USMLE", split="train")
    print("\nTrain example:")
    print(train_ds[0])

    dev_ds = load_dataset("mkieffer/MedQA-USMLE", split="dev")
    print("\nDev example:")
    print(dev_ds[0])

    test_ds = load_dataset("mkieffer/MedQA-USMLE", split="test")
    print("\nTest example:")
    print(test_ds[0])

    us_qbank_ds = load_dataset("mkieffer/MedQA-USMLE", split="us_qbank")
    print("\nUS QBank example:")
    print(us_qbank_ds[0])