import pandas as pd


def ru_persona_chat_dataset_tranformer_v1(
    initial_dataset_path: str,
    output_folder: str,
) -> None:
    """
        example
            ru_persona_chat_dataset_tranformer_v1(
            initial_dataset_path="./datasets/ru_persona_chat/dialogues.tsv",
            output_folder="./datasets/ru_persona_chat",
    )
    """
    assert initial_dataset_path is not None, "initial_dataset_path is None"
    assert output_folder is not None, "output_folder is None"

    dataset = pd.read_csv(initial_dataset_path, sep="\t")
    split_ratio = int(len(dataset) * 0.95)
    train_dataset = dataset[:split_ratio]
    valid_dataset = dataset[split_ratio:]

    print(f"Dataset lengths: train {len(train_dataset)}, valid {len(valid_dataset)}")
    # save csv files
    train_dataset.to_csv(output_folder + "/train.csv", index=False)
    valid_dataset.to_csv(output_folder + "/valid.csv", index=False)
    print("Datasets saved.")
