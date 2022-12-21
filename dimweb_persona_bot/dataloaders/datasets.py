from typing import List, Dict, Any, TypedDict, Optional
from abc import ABC, abstractmethod
import json

import pandas as pd


class AbstractInitialDataset(ABC):
    @abstractmethod
    def _build_dataset(
        self,
    ) -> None:
        pass

    @abstractmethod
    def _read_dataset(self, input_path: str) -> Dict:
        pass

    @abstractmethod
    def _create_initial_dataset(
        self,
        initial_dataset: List[Dict],
    ) -> List[Any]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class BaseDialogSampleV1(TypedDict):
    """
    context - это диалог, для этих данных порядок важен
    knowledge - это знания, например персона,
        порядок не важен, можно перемешивать
    sample_id - это id примера, обычно это просто порядковый номер
    dataset_source - откуда пришел пример, к примеру persona_chat
    """

    context: List[str]
    knowledge: List[str]
    sample_id: str
    dataset_source: str
    label: Optional[str]


class BaseInitialDatasetV1(AbstractInitialDataset):
    def __init__(self, input_dataset_path: str):
        assert input_dataset_path is not None, "input_dataset_path is None"

        self.input_dataset_path = input_dataset_path
        self.dataset = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        initial_dataset = self._read_dataset(self.input_dataset_path)
        self.dataset = self._create_initial_dataset(initial_dataset=initial_dataset)

    def _read_dataset(self, input_path: str) -> Dict:
        with open(input_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> BaseDialogSampleV1:
        return self.dataset[index]

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.dataset)
