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
from accelerate import Accelerator
import torch
import gc

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.1")


def h5_experiment_1():
    set_seed(2022)
    setup_gpus()
    model_name = "facebook/mbart-large-50"
    # model_name = "google/t5-efficient-tiny-nl8"
    # tokenizer_name = "./models/google-t5-efficient-tiny-nl8-ru"
    tokenizer_name = "facebook/mbart-large-50"

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        src_lang="ru_RU",
        tgt_lang="ru_RU",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    # dataset = load_from_disk("./datasets/ru_dialog_dataset_v1/")
    dataset = load_from_disk("./datasets/ru_dialog_dataset_v1_mbart-50/")
    dataset = dataset.remove_columns(
        [
            "label",
            "knowledge",
            "context",
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
        # print("all_columns", all_columns)

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
        run_id = wandb.run.id
        with open(f"./huggingface_training/{run_id}/result.txt", "a") as f:
            f.write("-" * 100 + "\n")
            f.write("-" * 50 + "EPOCH" + "-" * 50 + "\n")
            f.write("-" * 100 + "\n")
            for pred, label in zip(decoded_preds, decoded_labels):
                f.write(f"pred: {pred} label: {label}\n")

        return result

    deepspeed_config_zero3 = {
        "fp16": {
            "enabled": True,
            "loss_scale": 0.5,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 0.5,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 3,
            # "offload_optimizer": {
            #     "device": "cpu",
            #     "pin_memory": True,
            # },
            # "offload_param": {
            #     "device": "cpu",
            #     "pin_memory": True,
            # },
            # "offload_optimizer": {
            #     "device": "nvme",
            #     "nvme_path": "./local_nvme0",
            #     "pin_memory": True,
            #     "buffer_count": 4,
            #     "fast_init": False,
            # },
            # "offload_param": {
            #     "device": "nvme",
            #     "nvme_path": "./local_nvme1",
            #     "pin_memory": True,
            #     "buffer_count": 5,
            #     "buffer_size": 276055296,
            #     "max_in_cpu": 1e9,
            # },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": 1,
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
        "gradient_clipping": 1.0,
    }

    deepspeed_config_zero2 = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    # Initialize our Trainer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = "persona_bot_2"
    os.environ["WANDB_TAGS"] = "ru_dialog_dataset_v1,seq2seq_modeling,hypothesis_5"
    os.environ["WANDB_NAME"] = model_name
    wandb.init()
    run_id = wandb.run.id
    args = Seq2SeqTrainingArguments(
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        report_to="wandb",
        output_dir=f"./huggingface_training/{run_id}/",
        per_device_train_batch_size=4,
        per_gpu_eval_batch_size=16,
        logging_strategy="steps",
        logging_steps=20000,
        save_steps=20000,
        seed=2022,
        # fp16=True,
        # fp16_opt_level="O2",
        # fp16_full_eval=True,
        # fp16_backend="auto",
        # half_precision_backend="auto",
        dataloader_num_workers=16,
        run_name=model_name,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        # deepspeed=deepspeed_config_zero2,
        # deepspeed=deepspeed_config_zero3,
        # jit_mode_eval=True,
        # include_inputs_for_metrics=True,
    )

    train_dataloader, eval_dataloader = dataset["train"], dataset["test"]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.evaluate(eval_dataset=eval_dataloader)
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    wandb.finish()


if __name__ == "__main__":
    h5_experiment_1()
