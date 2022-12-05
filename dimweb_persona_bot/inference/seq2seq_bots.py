from typing import List
import random
import os

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
        answer = self.generate_response(sample)
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

        answer = self.generate_response(sample)
        answer = self.tokenizer.batch_decode(
            answer,
            skip_special_tokens=True,
        )
        return answer[0]

    def next_response(self) -> str:
        """
        делает предсказание на основе текущей истории
        и персоны

        полезно если мы управляем и отслеживаем состояние извне
        а этот бот нужен только для генерации ответов
        """

        sample = self._get_sample(
            persona=self.persona,
            history=self.history,
        )
        answer = self.generate_response(sample)
        answer = self.tokenizer.batch_decode(
            answer,
            skip_special_tokens=True,
        )
        self.history.append(answer[0])
        return answer[0]

    def generate_response(self, sample):
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

    def generate_response(self, sample: H2Seq2SeqInferenceSampleDictV2):
        return self.model.generate(
            **sample,
            max_new_tokens=60,
            penalty_alpha=0.3,
            top_k=20,
            top_p=0.95,
        )


def chat_with_bot_v2_console():
    if os.getlogin() != "dimweb":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        cuda_devices = ",".join(open("./cuda_devices", "r").read().split(" "))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    model_name = "./models/2vabb4b2/"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda:0"
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hyperparameters = H2PersonaChatHyperparametersV1(
        model_architecture="seq2seq",
        chat_history_pair_length=3,
        persona_max_length=31,
        chat_max_length=166,
    )

    bot2 = DialogBotV2(
        model=model,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        history=[],
        persona=[
            "Я студент 4 курса.",
            "Люблю иногда бухать по пятницам.",
            "У меня нет девушки.",
            "Люблю писать код.",
        ],
    )
    bot2.start_chat()


if __name__ == "__main__":
    chat_with_bot_v2_console()
