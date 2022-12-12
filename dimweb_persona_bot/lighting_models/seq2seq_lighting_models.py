from typing import List


from dimweb_persona_bot.lighting_models.causal_lighting_models import (
    LightingCausalModelV1,
)
from dimweb_persona_bot.database_logger.logger import DatabaseLoggerV1

import wandb
import os


class LightingSeq2SeqModelV1(LightingCausalModelV1):
    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        self.create_database_logger()

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

        decoded_labels = self.tokenizer.batch_decode(
            batch["custom_labels"],
            skip_special_tokens=True,
        )

        # compute text metrics
        text_metrics = self.text_evaluator.evaluate(
            generated_texts=generated_tokens,
            original_texts=decoded_labels,
        )

        # save texts for later analysis
        self.save_generation_predicts(
            prediction_ids=batch["sample_id"],
            decoded_labels=decoded_labels,
            generated_tokens=generated_tokens,
            input_tokens=input_tokens,
            persona=batch["persona"],
        )

        loss = predicts.loss

        self.log_dict(
            {
                "valid_loss": loss.detach().item(),
                "valid_blue_score": text_metrics["blue_score"],
                "valid_rougeL_score": text_metrics["rougeL_score"],
                "valid_chrf_score": text_metrics["chrf_score"],
            },
            on_step=True,
            on_epoch=True,
        )
