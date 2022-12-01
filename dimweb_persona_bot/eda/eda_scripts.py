from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import PersonaChatDatasetV2
from transformers import AutoTokenizer
import numpy as np
import pandas as pd


class TokensLengthAnalyzerV1:
    def __init__(
        self,
        train_dataset_path: str,
        valid_dataset_path: str,
    ):
        self.train_dataset = PersonaChatDatasetV2(
            input_dataset_path=train_dataset_path,
        )
        self.valid_dataset = PersonaChatDatasetV2(
            input_dataset_path=valid_dataset_path,
        )

        self.tokens_length_table = []
        self.fields_length_table = []

    def count_tokens(
        self,
        field: str,
        model: str,
        stage: str,
        dataset: PersonaChatDatasetV2,
        tokenizer: AutoTokenizer,
    ):
        lengths = []
        items_set = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            for item in sample[field]:
                items_set.add(item)

        items_set = list(items_set)
        for item in items_set:
            tokens = tokenizer.encode(item, add_special_tokens=False)
            tokens_len = len(tokens)
            lengths.append(tokens_len)

        percentile_95 = np.percentile(np.array(lengths), 95)

        self.tokens_length_table.append(
            {
                "field": field,
                "model": model,
                "95%": percentile_95,
                "stage": stage,
            }
        )

    def analyze(
        self,
        models_list: list = [
            "gpt2",
            "microsoft/DialoGPT-medium",
            "RUCAIBox/mvp",
            "roberta-base",
            "facebook/blenderbot_small-90M",
            "facebook/bart-base",
            "google/bigbird-pegasus-large-arxiv",
        ],
        stages_list: list = ["train", "valid"],
        fields_list: list = ["persona", "history"],
        save_path: str = "",
    ):
        assert save_path != "", "Please, specify save path"

        for model in models_list:
            print(f"Model: {model}")
            tokenizer = AutoTokenizer.from_pretrained(model)
            for stage in stages_list:
                dataset = self.train_dataset if stage == "train" else self.valid_dataset
                for field in fields_list:
                    self.count_tokens(
                        field=field,
                        model=model,
                        stage=stage,
                        dataset=dataset,
                        tokenizer=tokenizer,
                    )

        pd.DataFrame(self.tokens_length_table).to_csv(save_path, index=False)

    def show_info(
        self,
        dataset_path: str = "./dimweb_persona_bot/EDA/tokens_length_causal_models.csv",
        fields_list: list = ["persona", "history"],
        stages_list: list = ["train", "valid"],
        models_list: list = [
            "gpt2",
            "microsoft/DialoGPT-medium",
            "RUCAIBox/mvp",
            "roberta-base",
            "facebook/blenderbot_small-90M",
            "facebook/bart-base",
            "google/bigbird-pegasus-large-arxiv",
        ],
    ):
        tokens_length_causal_models = pd.read_csv(dataset_path)

        for field in fields_list:
            for stage in stages_list:
                print(f"Field: {field}, Stage: {stage}")
                average_length = tokens_length_causal_models[
                    (tokens_length_causal_models.field == field)
                    & (tokens_length_causal_models.stage == stage)
                ]["95%"].mean()
                print(f"Average replica length: {average_length}")
                print("\n")

        for field in fields_list:
            lengths = []
            for stage in stages_list:
                dataset = self.train_dataset if stage == "train" else self.valid_dataset
                for sample in dataset:
                    lengths.append(len(sample[field]))
            print(f"Field: {field}")
            print(f"Field len percentile 50: {np.percentile(np.array(lengths), 50)}")
            print(f"Field len percentile 95: {np.percentile(np.array(lengths), 95)}")

        tokenizer_lengths = []
        for model_name in models_list:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_len = tokenizer.model_max_length
            tokenizer_lengths.append(model_len)
            print(f"Model: {model_name}, Max length: {model_len}")
