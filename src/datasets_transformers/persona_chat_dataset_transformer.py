import json


def persona_chat_dataset_tranformer_v1(
    initial_dataset_path: str,
    output_folder: str,
) -> None:
    """
        example
            persona_chat_dataset_tranformer_v1(
            initial_dataset_path="./datasets/persona_chat/persona_chat.json",
            output_folder="./datasets/persona_chat",
    )
    """
    assert initial_dataset_path is not None, "initial_dataset_path is None"
    assert output_folder is not None, "output_folder is None"

    with open(initial_dataset_path) as f:
        initial_dataset = json.load(f)

    train_dataset = initial_dataset["train"]
    val_len = len(initial_dataset["valid"])
    valid_dataset = initial_dataset["valid"][: val_len // 2]
    test_dataset = initial_dataset["valid"][val_len // 2 :]

    print(
        f"Dataset lengths: train {len(train_dataset)}, valid {len(valid_dataset)}, test {len(test_dataset)}"
    )
    # save json files
    with open(output_folder + "/train.json", "w") as f:
        json.dump(train_dataset, f)

    with open(output_folder + "/valid.json", "w") as f:
        json.dump(valid_dataset, f)

    with open(output_folder + "/test.json", "w") as f:
        json.dump(test_dataset, f)

    print("Datasets saved.")
