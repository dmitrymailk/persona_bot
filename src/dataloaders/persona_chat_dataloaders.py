from typing import List, Dict, TypedDict

from src.dataloaders.datasets import BaseInitialDatasetV1


class PersonaChatDatasetSampleV1(TypedDict):
    """
    persona: List[str] - набор предложений фактов персоны
    history: List[str] - набор предложений истории переписки
    """

    persona: List[str]
    history: List[str]


class PersonaChatDatasetV1(BaseInitialDatasetV1):
    """
    датасет общего назначения который предоставляет
    интерфейс к оригинальному датасету никак не модифицируя его
    """

    def _create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[PersonaChatDatasetSampleV1]:
        dataset = []
        for item in initial_dataset:
            persona = item["personality"]
            utterances = item["utterances"]
            for utterance in utterances:
                history = utterance["history"]
                dataset_item = PersonaChatDatasetSampleV1(
                    persona=persona,
                    history=history,
                )
                dataset.append(dataset_item)

        return dataset

    def __getitem__(self, index: int) -> PersonaChatDatasetSampleV1:
        return self.dataset[index]
