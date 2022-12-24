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
        
        self.sep = " <s> "
        self.knowledge_prefix = " <k> "
        self.context_prefix = "<c> "
        # чисто интуитивно мне кажется что 4 это оптимальное число
        # для количества пар диалога
        self.dialog_pair_length = 4
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
        new_dataset = []
        for item in self.dataset:
            item: BaseDialogSampleV1
            new_context = self._prepare_context(item['context'])
            new_knowledge = self._prepare_knowledge(item['knowledge'])
            new_dataset.append(
                {
                    "context": new_context,
                    "knowledge": new_knowledge,
                    "dataset_source": item['dataset_source'],
                    "label": item['label'],
                    "sample_id": item['sample_id'],
                }
            )
        return pd.DataFrame(new_dataset)
    
    def _prepare_context(self, context: List[str]) -> str:
        context = self.sep.join(context)
        context = self.context_prefix + context
        return context
    
    def _prepare_knowledge(self, knowledge: List[str]) -> str:
        knowledge = self.sep.join(knowledge)
        knowledge = self.knowledge_prefix + knowledge
        return knowledge
