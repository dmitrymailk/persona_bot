from typing import Dict, TypedDict, List
import random

from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_1 import (
    BaseDatasetSampleV1,
)
from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_2 import (
    H2CausalTrainPersonaSampleV1,
)
from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_1 import (
    H1CausalSampleDictV1,
)
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_1 import (
    H1Seq2SeqSampleDictV2,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)
from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H4PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.utils import flat_list

from transformers import AutoTokenizer


class H4Seq2SeqTrainPersonaSampleV1(H2CausalTrainPersonaSampleV1):
    """
    Seq2Seq:
    входная последовательность:
    <bos> <persona> persona_fact[0]persona_fact[1]persona_fact[2]persona_fact[3]persona_fact[4]<sep>реплика[-6] реплика[-5] ... <query>реплика[-2]<query/><eos>
    таргет:<bos><response>реплика[-1]<response/><eos>

    <sep> - специальный токен, раздедяющий токен
    <query> - специальный токен, который оборачивает последнюю реплику пользователя
    <query/> -
    <response> - специальный токен, оборачивает ответ пользователя
    <response/>
    """

    def __init__(
        self,
        dataset_sample: PersonaChatDatasetSampleV1,
        tokenizer: AutoTokenizer,
        hyperparameters: H4PersonaChatHyperparametersV1,
    ) -> None:
        self.dataset_sample = dataset_sample
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters

        self.sep_token_id = self._get_token_id(self.hyperparameters.sep_token)
        self.query_token_open_id = self._get_token_id(
            self.hyperparameters.query_token_open
        )
        self.query_token_close_id = self._get_token_id(
            self.hyperparameters.query_token_close
        )
        self.response_token_open_id = self._get_token_id(
            self.hyperparameters.response_token_open
        )
        self.response_token_close_id = self._get_token_id(
            self.hyperparameters.response_token_close
        )

    def get_sample(self) -> H1CausalSampleDictV1:
        persona = self.dataset_sample["persona"]
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        # random.shuffle(persona)

        answer = history[-1]
        query = history[-2]
        dialog_history = history[:-2]

        encoded_persona = self._tok(
            persona,
            max_length=self.hyperparameters.persona_max_length,
        )

        encoded_query = self._tok(
            [query],
            max_length=self.hyperparameters.chat_max_length,
        )

        encoded_dialog_history = self._tok(
            dialog_history,
            max_length=self.hyperparameters.chat_max_length,
        )

        encoded_answer = self._tok(
            [answer],
            max_length=self.hyperparameters.chat_max_length,
        )

        input_ids = [
            *self.bos_token_id,
            *encoded_persona,
            self.sep_token_id,
            *encoded_dialog_history,
            self.query_token_open_id,
            *encoded_query,
            self.query_token_close_id,
            self.tokenizer.eos_token_id,
        ]

        labels = [
            *self.bos_token_id,
            self.response_token_open_id,
            *encoded_answer,
            self.response_token_close_id,
            self.tokenizer.eos_token_id,
        ]
        attention_mask = [1] * len(input_ids)

        return H1CausalSampleDictV1(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def _tok(
        self,
        text: List[str],
        max_length: int,
    ) -> List[int]:
        if len(text) == 0:
            return []

        tokens = self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        tokens = flat_list(tokens["input_ids"])
        return tokens

    @property
    def bos_token_id(self) -> List:
        bos_token = []
        if not "t5" in self.hyperparameters.model_name:
            if self.tokenizer.bos_token is not None:
                bos_token = [self.tokenizer.bos_token_id]

        return bos_token


class H4Seq2SeqValidPersonaSampleV1(H4Seq2SeqTrainPersonaSampleV1):
    def get_sample(self) -> H1CausalSampleDictV1:
        persona = self.dataset_sample["persona"]
        sample_id = self.dataset_sample["sample_id"]

        train_sample = super().get_sample()

        return H1Seq2SeqSampleDictV2(
            input_ids=train_sample["input_ids"],
            labels=train_sample["labels"],
            custom_labels=train_sample["labels"],
            attention_mask=train_sample["attention_mask"],
            sample_id=sample_id,
            persona=persona,
        )
