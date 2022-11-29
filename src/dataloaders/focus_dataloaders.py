from typing import List, Dict

from src.dataloaders.datasets import BaseInitialDatasetV1
from src.dataloaders.persona_chat_dataloaders import PersonaChatDatasetSampleV1


class FoCusDatasetV1(BaseInitialDatasetV1):
    def _create_initial_dataset(
        self,
        initial_dataset: Dict,
    ) -> List[PersonaChatDatasetSampleV1]:
        dataset = []
        initial_dataset_data = initial_dataset["data"]

        for dialog_set in initial_dataset_data:
            persona = dialog_set["persona"]
            utterances = dialog_set["utterance"]
            dialog_id = dialog_set["dialogID"]

            for utterance_pos, utterance in enumerate(utterances):
                knowledge_candidates = utterance["knowledge_candidates"]
                knowledge_answer_index = utterance["knowledge_answer_index"]
                dialog_index_key = [
                    item for item in utterance.keys() if "dialog" in item
                ][0]
                dialog = utterance[dialog_index_key]
                knowledge_candidate = knowledge_candidates[knowledge_answer_index]
                persona.append(knowledge_candidate)
                sample_id = f"{dialog_id}_{dialog_index_key}"
                data_sample = PersonaChatDatasetSampleV1(
                    persona=persona,
                    history=dialog,
                    sample_id=sample_id,
                )
                dataset.append(data_sample)

        return dataset
