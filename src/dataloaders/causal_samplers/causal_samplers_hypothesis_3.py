from typing import Dict, TypedDict, List
import random

from src.dataloaders.causal_samplers.causal_samplers_hypothesis_2 import (
    H2CausalTrainPersonaSampleV1,
)
from src.dataloaders.causal_samplers.causal_samplers_hypothesis_1 import (
    H1CausalSampleDictV1,
)


class H3CausalTrainPersonaSampleV1(H2CausalTrainPersonaSampleV1):
    """
    hypothesis_3.
    так же как и в hypothesis_2, но перемешиваем персону.
    """

    def get_sample(self) -> H1CausalSampleDictV1:
        persona = self.dataset_sample["persona"]
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
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

        input_ids = [
            self.tokenizer.bos_token_id,
            self.persona_id,
            *encoded_persona,
            self.chat_id,
            *encoded_history,
            self.tokenizer.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return H1CausalSampleDictV1(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_mask,
        )
