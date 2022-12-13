from typing import TypedDict, List

from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H2PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_1 import (
    H1CausalTrainPersonaSampleV1,
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


class H1Seq2SeqTrainPersonaSampleV1(H1CausalTrainPersonaSampleV1):
    """
    input_ids: all persona + history
    labels: user response
    """

    def add_spaces_between(self, items: List[str]) -> List[str]:
        items = self.add_spaces_after(items)
        items[-1] = items[-1].strip()
        return items

    def __init__(
        self,
        dataset_sample: PersonaChatDatasetSampleV1,
        tokenizer: AutoTokenizer,
        hyperparameters: H2PersonaChatHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters

    def get_sample(self) -> H1Seq2SeqSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        labels = [history.pop()]
        history = self.add_spaces_between(history)
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

        encoded_labels = self.tokenizer.batch_encode_plus(
            labels,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )

        encoded_labels = flat_list(encoded_labels["input_ids"])

        input_ids = [
            *self.bos_token_id,
            *encoded_persona,
            *encoded_history,
            *self.eos_token_id,
        ]
        labels = [
            *self.bos_token_id,
            *encoded_labels,
            *self.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return H1Seq2SeqSampleDictV1(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


class H1Seq2SeqValidPersonaSampleV1(H1Seq2SeqTrainPersonaSampleV1):
    """
    input_ids: all persona + history
    labels: user response
    """

    def get_sample(self) -> H1Seq2SeqSampleDictV2:
        persona = self.dataset_sample["persona"]
        persona = self.add_spaces_after(persona)
        sample_id = self.dataset_sample["sample_id"]

        sample = super().get_sample()

        return H1Seq2SeqSampleDictV2(
            input_ids=sample["input_ids"],
            labels=sample["labels"],
            custom_labels=sample["labels"],
            attention_mask=sample["attention_mask"],
            sample_id=sample_id,
            persona=persona,
        )
