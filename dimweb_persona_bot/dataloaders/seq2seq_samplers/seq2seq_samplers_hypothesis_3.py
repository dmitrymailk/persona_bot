import random
from typing import TypedDict, List

from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H1PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)
from dimweb_persona_bot.utils import flat_list

from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_2 import (
    H2Seq2SeqTrainPersonaSampleV1,
)
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_1 import (
    H1Seq2SeqSampleDictV1,
    H1Seq2SeqSampleDictV2,
)


class H3Seq2SeqTrainPersonaSampleV1(H2Seq2SeqTrainPersonaSampleV1):
    """
    seq2seq hypothesis_3

    """

    def get_sample(self) -> H1Seq2SeqSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        labels = history.pop()
        persona = self.dataset_sample["persona"]

        # shuffle persona
        random.shuffle(persona)

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

        encoded_labels = self.tokenizer.batch_encode_plus(
            [labels],
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )

        encoded_labels = flat_list(encoded_labels["input_ids"])

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

        labels = [
            *bos_token,
            *encoded_labels,
            self.tokenizer.eos_token_id,
        ]

        return H1Seq2SeqSampleDictV1(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
