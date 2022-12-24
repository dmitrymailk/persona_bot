from dimweb_persona_bot.dataloaders.datasets import (
    BaseInitialDatasetV1,
    BaseDialogSampleV1,
)
from typing import List


class RURubq20DatasetV1(BaseInitialDatasetV1):
    def _create_initial_dataset(self, initial_dataset) -> List[BaseDialogSampleV1]:
        dataset = []

        for dialog_id, question in enumerate(initial_dataset["questions"]):

            info_paragraph_id = question["paragraphs_uids"]["with_answer"]
            info_paragraph = ""
            if len(info_paragraph_id) > 0:
                info_paragraph_id = info_paragraph_id[0]

                info_paragraph = initial_dataset["paragraphs"].get(
                    str(info_paragraph_id), None
                )
                if info_paragraph is not None:
                    info_paragraph = info_paragraph["text"]

            dataset.append(
                BaseDialogSampleV1(
                    context=[question["question_text"]],
                    knowledge=[info_paragraph],
                    sample_id=f"RURubq20V1_{dialog_id}",
                    label=question["answer_text"],
                    dataset_source="RURubq20V1",
                )
            )

        return dataset
