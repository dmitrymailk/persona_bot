import os

from dimweb_persona_bot.dataloaders.ru_persona_chat_dataloaders import (
    RUPersonaChatDatasetV1,
)
from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_2 import (
    H2CausalValidPersonaSampleV1,
)
from dimweb_persona_bot.dataloaders.causal_samplers.causal_samplers_hypothesis_3 import (
    H3CausalTrainPersonaSampleV1,
)
from dimweb_persona_bot.dataloaders.lighting import LightningDataModuleV1
from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H2PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.lighting_models.causal_lighting_models import (
    LightingCausalModelV1,
)
from dimweb_persona_bot.hyperparameters.lighting import H1LightingHyperparametersV1
from dimweb_persona_bot.utils import (
    ExperimentArgumentParserV1,
    TrainArgumentsV1,
    WandbLoggerV2,
)

from transformers import AutoTokenizer, AutoModelForCausalLM


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def h6_experiment_1():
    """
    модели у которых сдвиг токенов происходит внутри модели
    - sberbank-ai/rugpt3medium_based_on_gpt2
    """
    if os.getlogin() != "dimweb":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        cuda_devices = ",".join(open("./cuda_devices", "r").read().split(" "))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 4
    if args.debug_status == 1:
        max_epochs = 2

    lighting_hyperparameters = H1LightingHyperparametersV1(
        precision=16,
        # accumulate_grad_batches=3,
        max_epochs=max_epochs,
        devices=[args.cuda_device],
    ).__dict__

    hyperparameters = H2PersonaChatHyperparametersV1(
        train_batch_size=16,
        valid_batch_size=32,
        model_name="sberbank-ai/rugpt3medium_based_on_gpt2",
        predicted_texts_folder="./predicted_texts",
        debug_status=args.debug_status,
        chat_history_pair_length=3,
        persona_max_length=10,
        chat_max_length=40,
    )

    tokenizer = AutoTokenizer.from_pretrained(hyperparameters.model_name)
    special_tokens = [
        "<c_sep>",
        "<p_sep>",
        "<chat>",
        "<persona>",
        "<responce>",
    ]
    tokenizer.add_tokens(
        special_tokens,
        special_tokens=True,
    )
    # так надо. https://github.com/huggingface/transformers/issues/2630#issuecomment-684512764
    tokenizer.pad_token_id = tokenizer.eos_token_id

    accelerator = "gpu"
    if args.debug_status == 1:
        # accelerator = "cpu"
        accelerator = "gpu"

    device = "cuda" if accelerator == "gpu" else "cpu"

    notes = """"""
    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
        tags=[
            "causal_modeling",
            "hypothesis_6",
            "ru_persona_chat",
        ],
        notes=notes,
    )

    data_module = LightningDataModuleV1(
        train_path_dataset="./datasets/ru_persona_chat/train.csv",
        valid_path_dataset="./datasets/ru_persona_chat/valid.csv",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        base_train_dataset_class=RUPersonaChatDatasetV1,
        base_valid_dataset_class=RUPersonaChatDatasetV1,
        base_train_sample_class=H3CausalTrainPersonaSampleV1,
        base_valid_sample_class=H2CausalValidPersonaSampleV1,
        debug_status=args.debug_status,
        device=device,
    )

    base_model = AutoModelForCausalLM.from_pretrained(hyperparameters.model_name)
    base_model.resize_token_embeddings(len(tokenizer))

    model = LightingCausalModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        base_model=base_model,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        filename=f"{hyperparameters.model_name}" + "-{epoch:02d}-{epoch:.2f}",
    )

    trainer = Trainer(
        accelerator=accelerator,
        logger=wandb_logger.logger,
        callbacks=[checkpoint_callback],
        **lighting_hyperparameters,
    )
    if args.debug_status != 1:
        trainer.validate(model=model, dataloaders=data_module)

    trainer.fit(
        model,
        datamodule=data_module,
    )
