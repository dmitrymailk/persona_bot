from typing import List, Dict, TypedDict

from dimweb_persona_bot.dataloaders.datasets import (
    BaseInitialDatasetV1,
    BaseDialogSampleV1,
)
from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import (
    PersonaChatDatasetSampleV1,
)

import pandas as pd
import re
from bs4 import BeautifulSoup


class RUPersonaChatReplicaV1(TypedDict):
    text: str
    persona_class: str


class RUPersonaChatDatasetV1(BaseInitialDatasetV1):
    """
    датасет общего назначения который предоставляет
    интерфейс к оригинальному датасету никак не модифицируя его.
    """

    def _simple_filter(
        self,
        text: str,
    ) -> str:
        text = text.replace("Пользователь 1: ", "")
        text = text.replace("Пользователь 2: ", "")
        text = text.replace("<br />", " ")
        text = text.replace("<br/>", " ")
        return text

    def _extract_persona(
        self,
        persona: str,
    ) -> str:
        soup = BeautifulSoup(
            persona,
            features="html.parser",
        )
        text = "".join([str(item) for item in soup.contents])
        text = text.replace("<br/>", ". ")
        text = text.replace(".. ", ". ")
        text = re.sub(r"<span.*\">|</span>", "", text)
        return [(item + ".").strip() for item in text.split(".") if item.strip()]

    def _extract_history(
        self,
        dialogue: str,
    ) -> List[RUPersonaChatReplicaV1]:
        dialogue_history = []

        soup = BeautifulSoup(
            dialogue,
            features="html.parser",
        )
        replicas = soup.find_all("span")
        current_class = replicas[0].get("class")[0]

        # create history like in original dataset(persona_chat)
        current_text = ""
        for replica in replicas:
            if replica.get("class")[0] == current_class:
                text = "".join([str(item) for item in replica.contents])
                text = self._simple_filter(text)
                if text[-1].isalpha():
                    text += ". "
                else:
                    text += " "

                current_text += text
            else:
                current_text = current_text.strip()
                replica_obj = RUPersonaChatReplicaV1(
                    text=current_text,
                    persona_class=current_class,
                )
                dialogue_history.append(replica_obj)
                current_class = replica.get("class")[0]

                text = "".join([str(item) for item in replica.contents])
                text = self._simple_filter(text)
                current_text = self._simple_filter(text)

                if current_text[-1].isalpha():
                    current_text += ". "
                else:
                    current_text += " "

        return dialogue_history

    def _create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[PersonaChatDatasetSampleV1]:
        dataset = []
        for dialogue_id in range(len(initial_dataset)):
            dialogue = initial_dataset.iloc[dialogue_id]["dialogue"]
            history = self._extract_history(dialogue=dialogue)

            first_persona = history[0]["persona_class"]
            persona = []
            if first_persona == "participant_2":
                persona = self._extract_persona(
                    initial_dataset.iloc[dialogue_id]["persona_1_profile"]
                )
            else:
                persona = self._extract_persona(
                    initial_dataset.iloc[dialogue_id]["persona_2_profile"]
                )

            history = [item["text"] for item in history]
            history_len = len(history) // 2
            for i in range(1, history_len):
                sample_id = f"{dialogue_id}_{i}"
                dataset.append(
                    PersonaChatDatasetSampleV1(
                        persona=persona,
                        history=history[: i * 2],
                        sample_id=sample_id,
                    )
                )

        return dataset

    def __getitem__(self, index: int) -> PersonaChatDatasetSampleV1:
        return self.dataset[index]

    def _read_dataset(self, input_path: str) -> Dict:
        dataset = pd.read_csv(
            input_path,
        )
        return dataset


class RUPersonaChatDatasetV2(RUPersonaChatDatasetV1):
    """
    датасет для анализа. возвращаю только полные диалоги.
    """

    def _create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[PersonaChatDatasetSampleV1]:
        dataset = []
        for dialogue_id in range(len(initial_dataset)):
            dialogue = initial_dataset.iloc[dialogue_id]["dialogue"]
            history = self._extract_history(dialogue=dialogue)

            first_persona = history[0]["persona_class"]
            persona = []
            if first_persona == "participant_2":
                persona = self._extract_persona(
                    initial_dataset.iloc[dialogue_id]["persona_1_profile"]
                )
            else:
                persona = self._extract_persona(
                    initial_dataset.iloc[dialogue_id]["persona_2_profile"]
                )

            history = [item["text"] for item in history]
            history_len = len(history) // 2
            dataset.append(
                PersonaChatDatasetSampleV1(
                    persona=persona,
                    history=history[: history_len * 2],
                    sample_id=str(dialogue_id),
                )
            )

        return dataset


class RUPersonaChatDatasetV3(RUPersonaChatDatasetV1):
    """
    для чтения изначального датасета формата tsv
    и приведение к единому формату
    """

    def _read_dataset(self, input_path: str) -> Dict:
        dataset = pd.read_csv(
            input_path,
            sep="\t",
        )
        return dataset

    def _create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[BaseDialogSampleV1]:
        dataset = []
        for dialogue_id in range(len(initial_dataset)):
            dialogue = initial_dataset.iloc[dialogue_id]["dialogue"]
            history = self._extract_history(dialogue=dialogue)

            first_persona = history[0]["persona_class"]
            persona = []
            if first_persona == "participant_2":
                persona = self._extract_persona(
                    initial_dataset.iloc[dialogue_id]["persona_1_profile"]
                )
            else:
                persona = self._extract_persona(
                    initial_dataset.iloc[dialogue_id]["persona_2_profile"]
                )

            history = [item["text"] for item in history]
            history_len = len(history) // 2 + 1
            for i in range(1, history_len):
                sample_id = f"RUPersonaChatDatasetV3_{dialogue_id}_{i}"
                context = history[: i * 2]
                context = context[-self.dialog_pair_length * 2 :]
                label = context.pop()

                dataset.append(
                    BaseDialogSampleV1(
                        context=context,
                        knowledge=persona,
                        dataset_source="RUPersonaChatDatasetV3",
                        label=label,
                        sample_id=sample_id,
                    )
                )

        return dataset
