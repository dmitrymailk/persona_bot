from dimweb_persona_bot.dataloaders.datasets import BaseInitialDatasetV1, BaseDialogSampleV1
from typing import List
import pandas as pd

class RUParusDatasetV1(BaseInitialDatasetV1):
    def _create_initial_dataset(self, initial_dataset) -> List[BaseDialogSampleV1]:
        dataset = []

        for i in range(len(initial_dataset)):
            sample = initial_dataset.iloc[i]
            label = sample['label'] + 1
            column_name = f"choice{label}"
            idx = str(i)
            label = sample[column_name]
            
            dataset_sample =  BaseDialogSampleV1(
                context=[
                    sample['premise']
                ],
                knowledge=[""],
                sample_id=idx,
                dataset_source='RUParusDatasetV1',
                label=label,
            )
            dataset.append(dataset_sample)

        return dataset

    def _read_dataset(self, input_path: str) -> str:
        # small dataset
        return pd.read_csv(input_path, encoding='utf-8')