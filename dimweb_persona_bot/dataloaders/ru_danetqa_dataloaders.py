from dimweb_persona_bot.dataloaders.datasets import (
    BaseInitialDatasetV1,
    BaseDialogSampleV1,
)
from typing import List
import pandas as pd
from tqdm import tqdm


class RUDanetqaDatasetV1(BaseInitialDatasetV1):
    def _create_initial_dataset(self, initial_dataset) -> List[BaseDialogSampleV1]:
        dataset = []

        for i in tqdm(range(len(initial_dataset))):
            sample = initial_dataset.iloc[i]

            label_text = ""
            label = sample["label"]
            if label:
                label_text = "Да"
            else:
                label_text = "Нет"

            knowledge = sample["passage"]
            context = sample["question"]
            idx = f"RUDanetqaDatasetV1_{i}"

            dataset_sample = BaseDialogSampleV1(
                context=[context],
                knowledge=[knowledge],
                sample_id=idx,
                dataset_source="RUDanetqaDatasetV1",
                label=label_text,
            )
            dataset.append(dataset_sample)

        return dataset

    def _read_dataset(self, input_path: str) -> str:
        # small dataset
        return pd.read_csv(input_path, encoding="utf-8")
