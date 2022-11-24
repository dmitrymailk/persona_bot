from typing import Dict, TypedDict, List

from src.hyperparameters.causal_modeling_hyperparameters import (
    H1PersonaChatHyperparametersV1,
)
from src.dataloaders.persona_chat_dataloaders import PersonaChatDatasetSampleV1
from src.dataloaders.causal_samplers.hypothesis_1 import (
    BaseDatasetSampleV1,
    H1CausalSampleDictV1,
    H1CausalSampleDictV2,
)

from src.utils import flat_list

from transformers import AutoTokenizer


class H2CausalTrainPersonaSampleV1(BaseDatasetSampleV1):
    """
    hypothesis_2.
    не сдвигаем labels и не укорачиваем input_ids
    """

    def __init__(
        self,
        dataset_sample: PersonaChatDatasetSampleV1,
        tokenizer: AutoTokenizer,
        hyperparameters: H1PersonaChatHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters

    def get_sample(self) -> H1CausalSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        persona = self.dataset_sample["persona"]

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

        input_ids = [
            self.tokenizer.bos_token_id,
            *encoded_persona,
            *encoded_history,
            self.tokenizer.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return H1CausalSampleDictV1(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_mask,
        )


class H2CausalValidPersonaSampleV1(H2CausalTrainPersonaSampleV1):
    """
    hypothesis_2.
    не сдвигаем labels и не укорачиваем input_ids
    """

    def get_sample(self) -> H1CausalSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        labels = history.pop()
        persona = self.dataset_sample["persona"]
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
            [labels],
            add_special_tokens=False,
            truncation=True,
        )
        encoded_labels = flat_list(encoded_labels["input_ids"])

        input_ids = [
            self.tokenizer.bos_token_id,
            *encoded_persona,
            *encoded_history,
            # self.tokenizer.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)
        custom_labels = [
            # self.tokenizer.bos_token_id,
            *encoded_labels,
            self.tokenizer.eos_token_id,
        ]

        return H1CausalSampleDictV2(
            input_ids=input_ids,
            labels=input_ids,
            custom_labels=custom_labels,
            attention_mask=attention_mask,
            sample_id=sample_id,
            persona=persona,
        )
