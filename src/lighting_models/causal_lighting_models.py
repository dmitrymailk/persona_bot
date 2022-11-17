from src.hyperparameters.causal_modeling_hyperparameters import BaseHyperparametersV1

from pytorch_lightning import LightningModule

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup

import torch


class LightingCausalModelV1(LightningModule):
    def __init__(
        self,
        hyperparameters: BaseHyperparametersV1,
        tokenizer: AutoTokenizer,
        base_model: AutoModelForCausalLM,
    ) -> None:
        super().__init__()
        self.hparams.update(hyperparameters.__dict__)
        self.save_hyperparameters(ignore=["base_model"])

        self.hyperparameters = hyperparameters
        self.tokenizer = tokenizer

        self.model = base_model

        self.predicts = {
            "train": {},
            "valid": {},
        }

    def training_step(
        self,
        batch,
        batch_idx: int,
    ):

        predicts = self.model(
            **batch,
        )

        loss = predicts.loss

        self.log("train_loss_step", loss)

        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        predicts = self.model(
            **batch,
        )

        loss = predicts.loss

        self.log("valid_loss_step", loss)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyperparameters.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hyperparameters.weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hyperparameters.learning_rate,
            eps=self.hyperparameters.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hyperparameters.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self) -> None:
        pass
