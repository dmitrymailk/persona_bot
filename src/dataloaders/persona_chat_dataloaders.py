from typing import List, Dict, TypedDict

from src.dataloaders.datasets import BaseInitialDatasetV1


class PersonaChatDatasetSampleV1(TypedDict):
    """
    persona: List[str] - набор предложений фактов персоны
    history: List[str] - набор предложений истории переписки
    """

    persona: List[str]
    history: List[str]
    sample_id: str


class PersonaChatDatasetV1(BaseInitialDatasetV1):
    """
    датасет общего назначения который предоставляет
    интерфейс к оригинальному датасету никак не модифицируя его.
    достаю из поля utterances диалоги и кладу рядом с ними персону
    """

    def _create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[PersonaChatDatasetSampleV1]:
        dataset = []
        for dialogue_id, dialogue in enumerate(initial_dataset):
            persona = dialogue["personality"]
            utterances = dialogue["utterances"]
            for utterance_id, utterance in enumerate(utterances):
                history = utterance["history"]
                sample_id = f"{dialogue_id}_{utterance_id}"
                if len(history) % 2 == 1:
                    history.pop()
                if len(history) > 0:
                    dataset_item = PersonaChatDatasetSampleV1(
                        persona=persona,
                        history=history,
                        sample_id=sample_id,
                    )
                    dataset.append(dataset_item)

        return dataset

    def __getitem__(self, index: int) -> PersonaChatDatasetSampleV1:
        return self.dataset[index]


class PersonaChatDatasetV2(BaseInitialDatasetV1):
    """
    датасет общего назначения который предоставляет
    интерфейс к оригинальному датасету никак не модифицируя его.
    достаю из поля utterances самый последний диалог и
    кладу рядом с ним персону.
    данный датасет нужен для отладки длины данных в датасете
    """

    def _create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[PersonaChatDatasetSampleV1]:
        dataset = []
        for dialogue_id, dialogue in enumerate(initial_dataset):
            persona = dialogue["personality"]
            utterance = dialogue["utterances"][-1]
            utterance_id = len(dialogue["utterances"]) - 1
            history = utterance["history"]
            sample_id = f"{dialogue_id}_{utterance_id}"
            if len(history) % 2 == 1:
                history.pop()
            if len(history) > 0:
                dataset_item = PersonaChatDatasetSampleV1(
                    persona=persona,
                    history=history,
                    sample_id=sample_id,
                )
                dataset.append(dataset_item)

        return dataset

    def __getitem__(self, index: int) -> PersonaChatDatasetSampleV1:
        return self.dataset[index]
