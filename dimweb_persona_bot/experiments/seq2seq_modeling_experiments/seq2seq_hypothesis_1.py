from dimweb_persona_bot.dataloaders.persona_chat_dataloaders import PersonaChatDatasetV1
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_1 import (
    H1Seq2SeqTrainPersonaSampleV1,
    H1Seq2SeqValidPersonaSampleV1,
)
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_2 import (
    H2Seq2SeqTrainPersonaSampleV1,
    H2Seq2SeqValidPersonaSampleV1,
    H2Seq2SeqTrainPersonaSampleV2,
)
from dimweb_persona_bot.dataloaders.lighting import LightningDataModuleV1
from dimweb_persona_bot.hyperparameters.causal_modeling_hyperparameters import (
    H2PersonaChatHyperparametersV1,
)
from dimweb_persona_bot.hyperparameters.lighting import H1LightingHyperparametersV1
from dimweb_persona_bot.utils import (
    ExperimentArgumentParserV1,
    TrainArgumentsV1,
    WandbLoggerV2,
)
from dimweb_persona_bot.lighting_models.seq2seq_lighting_models import (
    LightingSeq2SeqModelV1,
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def h1_experiment_1():
    """
    - facebook/bart-base
    - t5-base - ошибка nan (устранена на fp32)
    - microsoft/GODEL-v1_1-base-seq2seq
    """
    parser = ExperimentArgumentParserV1()
    args: TrainArgumentsV1 = parser.args

    max_epochs = 6
    if args.debug_status == 1:
        max_epochs = 2
    devices = [args.cuda_device]
    lighting_hyperparameters = H1LightingHyperparametersV1(
        precision=16,
        devices=devices,
        max_epochs=max_epochs,
    ).__dict__

    hyperparameters = H2PersonaChatHyperparametersV1(
        train_batch_size=32,
        valid_batch_size=64,
        model_name="facebook/bart-base",
        predicted_texts_folder="./predicted_texts",
        debug_status=args.debug_status,
        model_architecture="seq2seq",
    )

    tokenizer = AutoTokenizer.from_pretrained(hyperparameters.model_name)

    accelerator = "gpu"

    device = "cuda" if accelerator == "gpu" else "cpu"

    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
        tags=[
            "seq2seq_modeling",
            "hypothesis_2_1",
            "persona_bot_2",
        ],
    )

    data_module = LightningDataModuleV1(
        train_path_dataset="./datasets/persona_chat/train.json",
        valid_path_dataset="./datasets/persona_chat/valid.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        base_train_dataset_class=PersonaChatDatasetV1,
        base_valid_dataset_class=PersonaChatDatasetV1,
        base_train_sample_class=H2Seq2SeqTrainPersonaSampleV2,
        base_valid_sample_class=H2Seq2SeqValidPersonaSampleV1,
        debug_status=args.debug_status,
        device=device,
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(hyperparameters.model_name)

    model = LightingSeq2SeqModelV1(
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        base_model=base_model,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_blue_score_epoch",
        mode="max",
        filename=f"{hyperparameters.model_name}"
        + "-{epoch:02d}-{valid_blue_score_epoch:.2f}",
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
