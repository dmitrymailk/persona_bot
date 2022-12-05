import os

from dimweb_persona_bot.dataloaders.ru_persona_chat_dataloaders import (
    RUPersonaChatDatasetV1,
)
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_2 import (
    H2Seq2SeqValidPersonaSampleV1,
)
from dimweb_persona_bot.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_3 import (
    H3Seq2SeqTrainPersonaSampleV1,
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


def h6_experiment_1():
    """
    - facebook/mbart-large-50
    - facebook/bart-base
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

    devices = [args.cuda_device]

    hyperparameters = H2PersonaChatHyperparametersV1(
        train_batch_size=8,
        valid_batch_size=16,
        # model_name="t5-small",
        model_name="facebook/mbart-large-50",
        model_architecture="seq2seq",
        predicted_texts_folder="./predicted_texts",
        debug_status=args.debug_status,
        chat_history_pair_length=3,
        persona_max_length=31,
        chat_max_length=166,
    )

    deterministic = True
    # fix cumsum error
    if hyperparameters.model_name in ["google/long-t5-tglobal-base"]:
        deterministic = False

    lighting_hyperparameters = H1LightingHyperparametersV1(
        precision=16,
        # accumulate_grad_batches=3,
        max_epochs=max_epochs,
        devices=devices,
        deterministic=deterministic,
    ).__dict__

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

    accelerator = "gpu"
    if args.debug_status == 1:
        # accelerator = "cpu"
        accelerator = "gpu"

    device = "cuda" if accelerator == "gpu" else "cpu"

    notes = """"""
    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
        tags=[
            "seq2seq_modeling",
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
        base_train_sample_class=H3Seq2SeqTrainPersonaSampleV1,
        base_valid_sample_class=H2Seq2SeqValidPersonaSampleV1,
        debug_status=args.debug_status,
        device=device,
    )

    base_model = AutoModelForSeq2SeqLM.from_pretrained(hyperparameters.model_name)
    base_model.resize_token_embeddings(len(tokenizer))

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
    checkpoint_path = "./persona_bot/2vabb4b2/checkpoints/facebook/mbart-large-50-epoch=03-epoch=3.00.ckpt"
    # trainer = Trainer(
    #     accelerator=accelerator,
    #     logger=wandb_logger.logger,
    #     callbacks=[checkpoint_callback],
    #     **lighting_hyperparameters,
    # )
    # if args.debug_status != 1:
    #     trainer.validate(model=model, dataloaders=data_module)

    # trainer.fit(
    #     model,
    #     datamodule=data_module,
    # )
    model = LightingSeq2SeqModelV1.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        base_model=base_model,
    )
    model_path = "./models/2vabb4b2/"
    model.model.save_pretrained(model_path)
    model.tokenizer.save_pretrained(model_path)
