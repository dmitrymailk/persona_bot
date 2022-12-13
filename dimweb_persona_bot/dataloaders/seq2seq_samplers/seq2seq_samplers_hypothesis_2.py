from typing import TypedDict, List

from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H1PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)
from dimweb_persona_bot.utils import flat_list
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_1 import (
    H1Seq2SeqTrainPersonaSampleV1,
)
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_1 import (
    H1Seq2SeqSampleDictV1,
    H1Seq2SeqSampleDictV2,
)

from transformers import AutoTokenizer

import torch


class H2Seq2SeqInferenceSampleDictV1(TypedDict):
    input_ids: List[int]
    attention_mask: List[int]


class H2Seq2SeqInferenceSampleDictV2(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class H2Seq2SeqTrainPersonaSampleV1(H1Seq2SeqTrainPersonaSampleV1):
    """
    hypothesis 2
    """

    def add_sep_beetween(self, items: List[str], sep=" EOS ") -> List[str]:
        for i in range(1, len(items), 2):
            items[i] = sep + items[i]

        return items

    def get_sample(self) -> H1Seq2SeqSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        labels = [history.pop()]
        history = self.add_sep_beetween(history)
        persona = self.dataset_sample["persona"]
        persona = self.add_spaces_between(persona)

        KNOWLEDGE_IDS = self.tokenizer.encode(
            " [KNOWLEDGE] ",
            add_special_tokens=False,
        )
        CONTEXT_IDS = self.tokenizer.encode(
            " [CONTEXT] ",
            add_special_tokens=False,
        )

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
        "i am good, i just got off work and tired, i have</s>"
        "[CONTEXT] hello what are doing today? [KNOWLEDGE] i read twenty books a year. i'm a stunt double as my second job. i only eat kosher. i was raised in a single parent household.</s>"
        encoded_labels = flat_list(encoded_labels["input_ids"])

        input_ids = [
            *self.bos_token_id,
            *CONTEXT_IDS,
            *encoded_history,
            *KNOWLEDGE_IDS,
            *encoded_persona,
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


class H2Seq2SeqValidPersonaSampleV1(H2Seq2SeqTrainPersonaSampleV1):
    """
    hypothesis 2
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


class H2Seq2SeqInferencePersonaSampleV1(H2Seq2SeqTrainPersonaSampleV1):
    def get_sample(self) -> H2Seq2SeqInferenceSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        persona = self.dataset_sample["persona"]

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.persona_max_length,
        )

        encoded_persona = self._add_sep_token_persona(
            input_ids=encoded_persona["input_ids"],
        )

        encoded_history = self.tokenizer.batch_encode_plus(
            history,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )
        encoded_history = self._add_sep_token_chat_train(
            input_ids=encoded_history["input_ids"],
        )

        bos_token = self.get_bos_token_id()

        input_ids = [
            *bos_token,
            self.persona_id,
            *encoded_persona,
            self.chat_id,
            *encoded_history,
            self.tokenizer.eos_token_id,
        ]

        attention_mask = [1] * len(input_ids)

        return H2Seq2SeqInferenceSampleDictV1(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
