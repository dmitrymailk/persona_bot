from dimweb_persona_bot.dataloaders.datasets import (
    BaseInitialDatasetV1,
    BaseDialogSampleV1,
)
from typing import List
from tqdm import tqdm


class RUAnekdotsDialogsDatasetV1(BaseInitialDatasetV1):
    def _create_initial_dataset(self, initial_dataset) -> List[BaseDialogSampleV1]:
        dataset = []

        for dialog_id, dialog in tqdm(enumerate(initial_dataset.split("\n\n\n\n"))):
            dialog = dialog.split("\n")
            dialog = [self.filter_sentences(sentence) for sentence in dialog]

            for i in range(1, len(dialog) // 2 + 1):
                context = dialog[: i * 2]
                context = context[-self.dialog_pair_length * 2 :]
                label = context.pop()
                sample = BaseDialogSampleV1(
                    context=context,
                    knowledge="",
                    sample_id=f"RUAnekdotsDialogsV1_{dialog_id}_0_{i}",
                    dataset_source="RUAnekdotsDialogsV1",
                    label=label,
                )
                dataset.append(sample)

            dialog.pop(0)
            for i in range(1, len(dialog) // 2 + 1):
                context = dialog[: i * 2]
                context[-self.dialog_pair_length * 2 :]
                label = context.pop()
                sample = BaseDialogSampleV1(
                    context=context,
                    knowledge="",
                    sample_id=f"RUAnekdotsDialogsV1_{dialog_id}_1_{i}",
                    dataset_source="RUAnekdotsDialogsV1",
                    label=label,
                )
                dataset.append(sample)

        return dataset

    def _read_dataset(self, input_path: str) -> str:
        # small dataset
        return open(input_path).read()

    def filter_sentences(self, sentence: str):
        sentence = sentence.replace("- ", "")
        return sentence
