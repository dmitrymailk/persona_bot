from typing import Dict, TypedDict, List, Union
import random

from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H2PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)
from dimweb_persona_bot.utils import flat_list

from transformers import AutoTokenizer


class BaseDatasetSampleV1:
    def __init__(self) -> None:
        raise NotImplementedError

    def get_sample(self) -> Union[Dict, TypedDict]:
        raise NotImplementedError


class H1CausalSampleDictV1(TypedDict):
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


class H1CausalSampleDictV2(TypedDict):
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]
    custom_labels: List[int]
    sample_id: str
    persona: str


class H1CausalTrainPersonaSampleV1(BaseDatasetSampleV1):
    """
    hypothesis_1.
    """

    def __init__(
        self,
        dataset_sample: PersonaChatDatasetSampleV1,
        tokenizer: AutoTokenizer,
        hyperparameters: H2PersonaChatHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters

    def add_spaces_after(
        self,
        items: List[str],
    ) -> List[str]:
        items = [item + " " for item in items]
        return items

    def get_sample(self) -> H1CausalSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        history = self.add_spaces_after(history)

        persona = self.dataset_sample["persona"]
        persona = self.add_spaces_after(persona)

        encoded_history = self.tokenizer.batch_encode_plus(
            history,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )
        encoded_history = flat_list(encoded_history["input_ids"])

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.persona_max_length,
        )

        encoded_persona = flat_list(encoded_persona["input_ids"])

        input_ids = [
            *self.bos_token_id,
            *encoded_persona,
            *encoded_history,
            *self.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return H1CausalSampleDictV1(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_mask,
        )

    @property
    def bos_token_id(self):
        if "t5" in self.hyperparameters.model_name:
            return []

        if self.tokenizer.bos_token_id is None:
            return []

        return [self.tokenizer.bos_token_id]

    @property
    def eos_token_id(self):
        if self.tokenizer.eos_token_id is None:
            return []

        return [self.tokenizer.eos_token_id]


class H1CausalValidPersonaSampleV1(H1CausalTrainPersonaSampleV1):
    """
    hypothesis 1.
    """

    def get_sample(self) -> H1CausalSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        history = self.add_spaces_after(history)

        labels = [history.pop()]
        persona = self.dataset_sample["persona"]
        persona = self.add_spaces_after(persona)
        sample_id = self.dataset_sample["sample_id"]

        encoded_history = self.tokenizer.batch_encode_plus(
            history,
            add_special_tokens=False,
            truncation=True,
        )
        encoded_history = flat_list(encoded_history["input_ids"])

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
        )

        encoded_persona = flat_list(encoded_persona["input_ids"])

        encoded_labels = self.tokenizer.batch_encode_plus(
            labels,
            add_special_tokens=False,
            truncation=True,
        )
        encoded_labels = flat_list(encoded_labels["input_ids"])

        input_ids = [
            *self.bos_token_id,
            *encoded_persona,
            *encoded_history,
            # self.tokenizer.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)
        custom_labels = [
            # self.tokenizer.bos_token_id,
            *encoded_labels,
            *self.eos_token_id,
        ]

        return H1CausalSampleDictV2(
            input_ids=input_ids,
            labels=input_ids,
            custom_labels=custom_labels,
            attention_mask=attention_mask,
            sample_id=sample_id,
            persona=persona,
        )


class H1CausalTrainPersonaSampleV2(H1CausalTrainPersonaSampleV1):
    """
    hypothesis 1.1
    """

    def get_sample(self) -> H1CausalSampleDictV1:
        random.shuffle(self.dataset_sample["persona"])
        return super().get_sample()


class H1CausalTrainPersonaSampleV3(H1CausalTrainPersonaSampleV1):
    """
    hypothesis 1.2
    """

    def get_sample(self) -> H1CausalSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        history = self.add_spaces_after(history)

        label = history.pop()
        random.shuffle(history)
        history.append(label)

        persona = self.dataset_sample["persona"]
        random.shuffle(persona)
        persona = self.add_spaces_after(persona)

        encoded_history = self.tokenizer.batch_encode_plus(
            history,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )
        encoded_history = flat_list(encoded_history["input_ids"])

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.persona_max_length,
        )

        encoded_persona = flat_list(encoded_persona["input_ids"])

        input_ids = [
            *self.bos_token_id,
            *encoded_persona,
            *encoded_history,
            *self.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return H1CausalSampleDictV1(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_mask,
        )
