import logging
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset, load_from_disk

import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    DataCollatorForSeq2Seq,
)
from transformers.utils import check_min_version

from dimweb_persona_bot.utils import TextEvaluator, setup_gpus

import wandb


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.1")


def h5_experiment_1():
    set_seed(2022)
    setup_gpus()
    model_name = "facebook/mbart-large-50"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    dataset = load_from_disk("./datasets/ru_dialog_dataset_v1/")
    dataset = dataset.remove_columns(
        [
            "dataset_source",
            "label",
            "sample_id",
            "input_ids",
            "attention_mask",
            "labels",
        ]
    )
    # Metric
    text_evaluator = TextEvaluator()

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
        )

        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id,
        )

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds,
            decoded_labels,
        )

        result = text_evaluator.evaluate(
            generated_texts=decoded_preds,
            original_texts=decoded_labels,
        )

        return result

    # Initialize our Trainer
    args = Seq2SeqTrainingArguments(
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        report_to="wandb",
        output_dir="./huggingface_training/",
        per_device_train_batch_size=16,
        per_gpu_eval_batch_size=32,
        logging_strategy="steps",
        logging_steps=10000,
        save_steps=10000,
        seed=2022,
        use_ipex=True,
        fp16_opt_level="O3",
        fp16_full_eval=True,
        # dataloader_num_workers=8,
        run_name=model_name,
        load_best_model_at_end=True,
        metric_for_best_model="blue_score",
        greater_is_better=True,
        dataloader_drop_last=True,
    )

    os.environ["WANDB_PROJECT"] = "persona_bot_2"
    os.environ["WANDB_TAGS"] = "ru_dialog_dataset_v1,seq2seq_modeling,hypothesis_5"

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        # data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.evaluate(eval_dataset=dataset["test"])
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    wandb.finish()
