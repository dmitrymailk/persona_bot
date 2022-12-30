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
from dimweb_persona_bot.utils import setup_gpus
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

        self.debug_status = hyperparameters.debug_status

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
        if self.debug_status == 1:
            print(f"PERSONA: {self.persona}")

        while True:
            message = input("You: ")

            if self.debug_status == 1:
                print("-" * 100)

            if message == "exit":
                break
            answer = self.chat(message)

            if self.debug_status:
                print("CONTEXT:", self.history)

            print("Bot:", answer)


class DialogBotV2(DialogBotV1):
    """
    bot uses Contrastive Search
    """

    def generate_response(self, sample: H2Seq2SeqInferenceSampleDictV2):
        return self.model.generate(
            **sample,
            max_new_tokens=60,
            penalty_alpha=0.15,
            top_k=10,
        )


class DialogBotV3:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        history: List[str] = None,
        device: str = "cuda",
        max_pairs: int = 3,
        max_response_length: int = 200,
        debug_status: int = 0,
        sep_token: str = " <sep> ",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.debug_status = debug_status
        self.max_pairs = max_pairs
        self.max_response_length = max_response_length
        self.sep_token = sep_token

        if history is None:
            self.history = []
        self.history = history

        self.last_response = ""

    def _get_sample(
        self,
        history: List[str],
    ):
        history = history.copy()
        last_response = [history.pop()]
        history = history[-self.max_pairs * 2 :]
        history = history + last_response

        history = self.tokenizer.batch_decode(
            self.tokenizer.batch_encode_plus(
                history,
                add_special_tokens=False,
                max_length=self.max_response_length,
                truncation=True,
            )["input_ids"],
            add_special_tokens=False,
        )

        if self.debug_status == 1:
            print(f"history: {history}")
            print(f"last_response: {last_response}")
            print(f"self.max_pairs: {self.max_pairs}")

        history = self.sep_token.join(history)
        print(history)
        sample = self.tokenizer(
            history,
            max_length=1024,
            return_tensors="pt",
            truncation=True,
        )

        for key in sample.keys():
            sample[key] = sample[key].to(self.device)

        return sample

    def chat(
        self,
        message: str,
    ) -> str:

        self.history.append(message)

        sample = self._get_sample(
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

        temp_history = self.history.copy()
        temp_history.append(message)

        sample = self._get_sample(
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
            history=self.history,
        )
        answer = self.generate_response(sample)
        answer = self.tokenizer.batch_decode(
            answer,
            skip_special_tokens=True,
        )

        return answer[0]

    def generate_response(self, sample):
        return self.model.generate(
            **sample,
            max_new_tokens=self.max_response_length,
            penalty_alpha=0.2,
            top_k=10,
            # min_length=8,
            top_p=0.98,
            do_sample=True,
        )

    def start_chat(self):

        while True:
            message = input("You: ")

            if self.debug_status == 1:
                print("-" * 100)

            if message == "exit":
                break
            answer = self.chat(message)

            if self.debug_status:
                print("CONTEXT:", self.history)

            if self.last_response == answer:
                self.history = []
            else:
                self.last_response = answer

            print("Bot:", answer)


def chat_with_model_persona_bot_2_28akcwik():
    setup_gpus()
    model_path = "dim/persona_bot_2_28akcwik"
    device = "cuda"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    hyperparameters = H2PersonaChatHyperparametersV1(
        chat_history_pair_length=7,
        persona_max_length=14,
        chat_max_length=25,
        debug_status=1,
    )

    bot2 = DialogBotV2(
        model=model,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
        history=[],
        persona=[
            "i'm also a graduate student .",
            "i enjoy reading journals and guides related to psychology .",
            "my parents taught me survival skills .",
            "i walk dogs for a living .",
        ],
    )

    bot2.start_chat()


def chat_with_model_ru_mbart_50():
    setup_gpus()
    path = "./models/2uuglxhm/checkpoint-440000"
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(path)

    ru_bot = DialogBotV3(
        model=model,
        tokenizer=tokenizer,
        history=[],
        debug_status=1,
        max_pairs=1,
    )

    ru_bot.start_chat()


if __name__ == "__main__":
    # python -m dimweb_persona_bot.inference.seq2seq_bots
    setup_gpus()
    # chat_with_model_persona_bot_2_28akcwik()
    chat_with_model_ru_mbart_50()
