import os

from src.dataloaders.persona_chat_dataloaders import PersonaChatDatasetV1
from src.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_2 import (
    H2Seq2SeqValidPersonaSampleV1,
)
from src.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_3 import (
    H3Seq2SeqTrainPersonaSampleV1,
)
from src.dataloaders.lighting import LightningDataModuleV1
from src.hyperparameters.causal_modeling_hyperparameters import (
    H2PersonaChatHyperparametersV1,
)
from src.lighting_models.causal_lighting_models import LightingCausalModelV1
from src.hyperparameters.lighting import H1LightingHyperparametersV1
from src.utils import (
    ExperimentArgumentParserV1,
    TrainArgumentsV1,
    WandbLoggerV2,
)
from src.lighting_models.seq2seq_lighting_models import LightingSeq2SeqModelV1

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torch


def h3_experiment_1():
    """
    - t5-small - ошибка nan (из-за fp16, устранена на fp32)
    - t5-base - ошибка nan (устранена на fp32)
    - facebook/bart-base
    - google/t5-v1_1-small
    - facebook/blenderbot-400M-distill
    - google/long-t5-tglobal-base
    - google/bigbird-pegasus-large-arxiv - не помещается на карту
    - allenai/led-base-16384 - очень долго обучается
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
        model_name="facebook/bart-base",
        model_architecture="seq2seq",
        predicted_texts_folder="./predicted_texts",
        debug_status=args.debug_status,
        chat_history_pair_length=3,
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

    notes = """
    дефолтная AutoModelForSeq2SeqLM.
    контекст=вся персона+последний вопрос от пользователя
    таргет=ответ от пользователя
    перемешиваю персону при тренировке. не перемешиваю при валидации.
    """
    wandb_logger = WandbLoggerV2(
        hyperparameters=hyperparameters,
        tags=[
            "seq2seq_modeling",
            "hypothesis_3",
        ],
        notes=notes,
    )

    data_module = LightningDataModuleV1(
        train_path_dataset="./datasets/persona_chat/train.json",
        valid_path_dataset="./datasets/persona_chat/valid.json",
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        base_train_dataset_class=PersonaChatDatasetV1,
        base_valid_dataset_class=PersonaChatDatasetV1,
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
