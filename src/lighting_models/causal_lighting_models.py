import os

from src.hyperparameters.causal_modeling_hyperparameters import BaseHyperparametersV1
from src.utils import TextEvaluator

from pytorch_lightning import LightningModule

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup

import torch

import wandb


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

        self.text_evaluator = TextEvaluator()

        self.model = base_model

        self.predicts = {
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

        self.log("train_loss", loss)

        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        predicts = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        generated_tokens = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.hyperparameters.max_response_length,
        )

        generated_tokens = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        input_tokens = self.tokenizer.batch_decode(
            batch["input_ids"],
            skip_special_tokens=True,
        )

        cut_generated_tokens = []
        for i, generated_token in enumerate(generated_tokens):
            cut_token = generated_token[len(input_tokens[i]) :]
            cut_generated_tokens.append(cut_token)

        decoded_labels = self.tokenizer.batch_decode(
            batch["custom_labels"],
            skip_special_tokens=True,
        )

        # compute text metrics
        text_metrics = self.text_evaluator.evaluate(
            generated_texts=cut_generated_tokens,
            original_texts=decoded_labels,
        )

        # save texts for later analysis
        run_id = wandb.run.id
        epoch = self.current_epoch
        paired_texts = []
        for label, generated_text, input_token in zip(
            decoded_labels,
            cut_generated_tokens,
            input_tokens,
        ):
            true_texts = f"{input_token}@@@{label}"
            predicted_texts = f"{input_token}@@@{generated_text}"

            true_texts = true_texts.replace("\n", " ")
            predicted_texts = predicted_texts.replace("\n", " ")

            pair = "\n".join([true_texts, predicted_texts])
            paired_texts.append(pair)

        paired_texts = "\n\n".join(paired_texts)

        save_folder_path = f"{self.hyperparameters.predicted_texts_folder}/{run_id}"
        # create folder if not exists
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        # append to file
        with open(f"{save_folder_path}/{epoch}.txt", "a") as f:
            f.write(paired_texts)

        loss = predicts.loss

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_blue_score": text_metrics["blue_score"],
                "valid_rougeL_score": text_metrics["rougeL_score"],
                "valid_chrf_score": text_metrics["chrf_score"],
            },
            on_step=True,
            on_epoch=True,
        )

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
