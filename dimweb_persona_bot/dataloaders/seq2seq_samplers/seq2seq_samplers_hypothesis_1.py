from typing import TypedDict, List

from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H1PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)
from dimweb_persona_bot.utils import flat_list
from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_1 import (
    BaseDatasetSampleV1,
)

from transformers import AutoTokenizer


class H1Seq2SeqSampleDictV1(TypedDict):
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


class H1Seq2SeqSampleDictV2(TypedDict):
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]
    custom_labels: List[int]
    sample_id: str
    persona: str


class H1Seq2SeqTrainPersonaSampleV1(BaseDatasetSampleV1):
    """
    input_ids: all persona + history + eos
    labels: user response + eos
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

    def get_sample(self) -> H1Seq2SeqSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        labels = history.pop()
        persona = self.dataset_sample["persona"]

        encoded_history = self.tokenizer.batch_encode_plus(
            history,
            add_special_tokens=False,
            truncation=True,
        )
        encoded_history = flat_list(encoded_history["input_ids"])
        encoded_history = encoded_history[:128]

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
        )

        encoded_persona = flat_list(encoded_persona["input_ids"])
        encoded_persona = encoded_persona[:128]

        encoded_labels = self.tokenizer.batch_encode_plus(
            [labels],
            add_special_tokens=False,
            truncation=True,
        )

        encoded_labels = flat_list(encoded_labels["input_ids"])

        encoded_task = self.tokenizer.batch_encode_plus(
            ["chat:"],
            add_special_tokens=False,
            truncation=True,
        )
        encoded_task = flat_list(encoded_task["input_ids"])

        bos_token = []
        if not "t5" in self.hyperparameters.model_name:
            if self.tokenizer.bos_token is not None:
                bos_token = [self.tokenizer.bos_token_id]

        input_ids = [
            *bos_token,
            *encoded_task,
            *encoded_persona,
            *encoded_history,
            self.tokenizer.eos_token_id,
        ]
        labels = [
            *bos_token,
            *encoded_labels,
            self.tokenizer.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return H1Seq2SeqSampleDictV1(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


class H1Seq2SeqValidPersonaSampleV1(H1Seq2SeqTrainPersonaSampleV1):
    """
    input_ids: all persona + history + eos
    labels: user response + eos
    """

    def get_sample(self) -> H1Seq2SeqSampleDictV2:
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
        encoded_history = encoded_history[:128]

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
        )

        encoded_persona = flat_list(encoded_persona["input_ids"])
        encoded_persona = encoded_persona[:128]

        encoded_labels = self.tokenizer.batch_encode_plus(
            [labels],
            add_special_tokens=False,
            truncation=True,
        )
        encoded_labels = flat_list(encoded_labels["input_ids"])

        encoded_task = self.tokenizer.batch_encode_plus(
            ["chat:"],
            add_special_tokens=False,
            truncation=True,
        )
        encoded_task = flat_list(encoded_task["input_ids"])

        bos_token = []
        if not "t5" in self.hyperparameters.model_name:
            if self.tokenizer.bos_token is not None:
                bos_token = [self.tokenizer.bos_token_id]

        input_ids = [
            *bos_token,
            *encoded_task,
            *encoded_persona,
            *encoded_history,
            self.tokenizer.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)
        custom_labels = [
            *bos_token,
            *encoded_labels,
            self.tokenizer.eos_token_id,
        ]
        labels = custom_labels

        return H1Seq2SeqSampleDictV2(
            input_ids=input_ids,
            labels=labels,
            custom_labels=custom_labels,
            attention_mask=attention_mask,
            sample_id=sample_id,
            persona=persona,
        )
