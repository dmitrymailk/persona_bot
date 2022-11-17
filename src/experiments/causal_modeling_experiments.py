from src.dataloaders.persona_chat_dataloaders import PersonaChatDatasetV1
from src.dataloaders.causal_samplers import CausalPersonaSampleV1
from src.dataloaders.lighting import LightningDataModuleV1
from src.hyperparameters.causal_modeling_hyperparameters import (
    PersonaChatHyperparametersV1,
)

from transformers import AutoTokenizer


def experiment_1():
    hyperparameters = PersonaChatHyperparametersV1(train_batch_size=2)
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters.model_name)

    lighting_data = LightningDataModuleV1(
        train_path_dataset="./datasets/persona_chat/train.json",
        valid_path_dataset="./datasets/persona_chat/valid.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        base_train_dataset_class=PersonaChatDatasetV1,
        base_valid_dataset_class=PersonaChatDatasetV1,
        base_train_sample_class=CausalPersonaSampleV1,
        base_valid_sample_class=CausalPersonaSampleV1,
    )
    lighting_data.setup()
    next(iter(lighting_data.train_dataloader()))
    print("end")


experiment_1()
