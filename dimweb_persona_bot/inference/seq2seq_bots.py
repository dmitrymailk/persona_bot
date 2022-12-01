from typing import List
import random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_2 import (
    H2Seq2SeqInferencePersonaSampleV1,
    H2Seq2SeqInferenceSampleDictV2,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)
from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H2PersonaChatHyperparametersV1,
)

import torch


class DialogBotV1:
    """
    bot uses greedy decoding
    """

    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        hyperparameters: H2PersonaChatHyperparametersV1,
        history: List[str] = None,
        persona: List[str] = None,
        device: str = "cuda",
        shuffle_persona: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hyperparameters = hyperparameters
        self.device = device
        self.shuffle_persona = shuffle_persona

        if history is None:
            self.history = []
        self.history = history

        if persona is None:
            self.persona = []
        self.persona = persona

    def _get_sample(
        self,
        persona: List[str],
        history: List[str],
    ) -> H2Seq2SeqInferenceSampleDictV2:
        dataset_sample = PersonaChatDatasetSampleV1(
            persona=persona,
            history=history,
        )

        sample = H2Seq2SeqInferencePersonaSampleV1(
            tokenizer=self.tokenizer,
            hyperparameters=self.hyperparameters,
            dataset_sample=dataset_sample,
        )
        sample = sample.get_sample()

        for key in sample.keys():
            sample[key] = torch.tensor(sample[key]).unsqueeze(0).to(self.device)

        return sample

    def chat(
        self,
        message: str,
    ) -> str:
        if self.shuffle_persona:
            random.shuffle(self.persona)

        self.history.append(message)

        sample = self._get_sample(
            persona=self.persona,
            history=self.history,
        )
        answer = self.generate_responce(sample)
        answer = self.tokenizer.batch_decode(
            answer,
            skip_special_tokens=True,
        )
        self.history.append(answer[0])
        return answer[0]

    def single_chat(
        self,
        message: str,
    ) -> str:
        if self.shuffle_persona:
            random.shuffle(self.persona)

        temp_history = self.history.copy()
        temp_history.append(message)

        sample = self._get_sample(
            persona=self.persona,
            history=temp_history,
        )

        answer = self.generate_responce(sample)
        answer = self.tokenizer.batch_decode(
            answer,
            skip_special_tokens=True,
        )
        return answer[0]

    def generate_responce(self, sample):
        return self.model.generate(**sample, max_length=20)

    def start_chat(self):
        while True:
            message = input("You: ")
            if message == "exit":
                break
            answer = self.chat(message)
            print("Bot:", answer)


class DialogBotV2(DialogBotV1):
    """
    bot uses Contrastive Search
    """

    def generate_responce(self, sample: H2Seq2SeqInferenceSampleDictV2):
        return self.model.generate(
            **sample,
            max_new_tokens=20,
            penalty_alpha=0.1,
            top_k=6,
        )
