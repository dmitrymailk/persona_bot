from typing import Dict, TypedDict, List

from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H2PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)
from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_1 import (
    BaseDatasetSampleV1,
    H1CausalSampleDictV1,
    H1CausalSampleDictV2,
)

from dimweb_persona_bot.utils import flat_list

from transformers import AutoTokenizer


class H2CausalTrainPersonaSampleV1(BaseDatasetSampleV1):
    """
    hypothesis_2.
    не сдвигаем labels и не укорачиваем input_ids

    входной пример будет выглядеть так:
        bos_token_id,
        persona_id,
        persona_1,
        persona_sep_token_id,
        persona_2,
        persona_sep_token_id,
        ...
        chat_id,
        chat_1,
        chat_sep_token_id,
        chat_2,
        ...
        chat_n,
                <- тут не идет chat_sep_token_id, а сразу завершение строки
        eos_token_id
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

        self.p_sep_id = self._get_token_id(self.hyperparameters.persona_sep_token)
        self.c_sep_id = self._get_token_id(self.hyperparameters.chat_sep_token)
        self.persona_id = self._get_token_id(self.hyperparameters.persona_token)
        self.chat_id = self._get_token_id(self.hyperparameters.chat_token)
        self.response_id = self._get_token_id(self.hyperparameters.responce_token)

    def _get_token_id(self, text: str) -> int:
        return self.tokenizer.encode(text, add_special_tokens=False)[0]

    def _add_sep_token_persona(
        self,
        input_ids: List[List[int]],
    ) -> List[int]:
        result = []
        for pos, item in enumerate(input_ids):
            result.extend(item)
            if pos != len(input_ids) - 1:
                result.append(self.p_sep_id)

        return result

    def _add_sep_token_chat_valid(
        self,
        input_ids: List[List[int]],
    ) -> List[int]:
        result = []
        for pos, item in enumerate(input_ids):
            result.extend(item)
            if pos < len(input_ids) - 1:
                result.append(self.c_sep_id)
            else:
                result.append(self.response_id)
        return result

    def _add_sep_token_chat_train(
        self,
        input_ids: List[List[int]],
    ) -> List[int]:
        result = []
        for pos, item in enumerate(input_ids):
            result.extend(item)
            if pos < len(input_ids) - 2:
                result.append(self.c_sep_id)
            if pos == len(input_ids) - 2:
                result.append(self.response_id)

        return result

    def get_sample(self) -> H1CausalSampleDictV1:
        persona = self.dataset_sample["persona"]
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]

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


class H2CausalValidPersonaSampleV1(H2CausalTrainPersonaSampleV1):
    """
    hypothesis_2.
    не сдвигаем labels и не укорачиваем input_ids.
    входной пример будет выглядеть так:
        bos_token_id,
        persona_id,
        persona_1,
        persona_sep_token_id,
        persona_2,
        persona_sep_token_id,
        ...
        chat_id,
        chat_1,
        chat_sep_token_id,
        chat_2,
        ...
        chat_n-1,
        chat_sep_token_id <- это важно, потому что после этого мы должны
        будем предсказать ответ и этот ответ должен содержать chat_n и eos_token_id
    """

    def get_sample(self) -> H1CausalSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        labels = history.pop()
        persona = self.dataset_sample["persona"]
        sample_id = self.dataset_sample["sample_id"]

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
        encoded_history = self._add_sep_token_chat_valid(
            input_ids=encoded_history["input_ids"],
        )

        encoded_labels = self.tokenizer.batch_encode_plus(
            [labels],
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )

        encoded_labels = flat_list(encoded_labels["input_ids"])

        input_ids = [
            self.tokenizer.bos_token_id,
            self.persona_id,
            *encoded_persona,
            self.chat_id,
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
