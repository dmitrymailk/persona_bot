import argparse
from dataclasses import dataclass
from itertools import chain
from typing import List, Dict
import os

from datasets import load_metric

from rouge_score import rouge_scorer

from torchmetrics import CHRFScore

from typing import List
from itertools import chain

from pytorch_lightning.loggers import WandbLogger


def flat_list(list_of_lists: List[List]) -> List:
    return list(chain.from_iterable(list_of_lists))


def setup_gpus():
    if os.getlogin() != "dimweb":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        cuda_devices = ",".join(open("./cuda_devices", "r").read().split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


class TextEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.bleu = load_metric("sacrebleu")
        self.chrf = CHRFScore()

    def evaluate(
        self,
        generated_texts: List[str],
        original_texts: List[str],
    ):
        blue_score = self.bleu.compute(
            predictions=generated_texts,
            references=[[item] for item in original_texts],
        )["score"]

        # compute rouge score
        rougeL_score = 0
        for gen_text, orig_text in zip(generated_texts, original_texts):
            scores = self.rouge.score(orig_text, gen_text)
            rougeL_score += scores["rougeL"].fmeasure

        rougeL_score /= len(generated_texts)

        # compute chrf score
        chrf_score = self.chrf(
            generated_texts, [[item] for item in original_texts]
        ).item()

        return {
            "blue_score": blue_score,
            "rougeL_score": rougeL_score,
            "chrf_score": chrf_score,
        }


@dataclass
class TrainArgumentsV1:
    debug_status: int
    cuda_device: int


class ExperimentArgumentParserV1:
    """Todo: сделать типизацию через наследование от Namespace"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="training arguments")
        params = [
            (
                "--debug_status",
                {
                    "dest": "debug_status",
                    "type": int,
                    "default": 0,
                },
            ),
            (
                "--cuda_device",
                {
                    "dest": "cuda_device",
                    "type": int,
                    "default": 0,
                },
            ),
        ]

        for name, param in params:
            parser.add_argument(name, **param)

        args = parser.parse_args()
        args = args._get_kwargs()
        args = {arg[0]: arg[1] for arg in args}

        args = TrainArgumentsV1(**args)

        self.args = args

        setup_gpus()


class WandbLoggerV1:
    def __init__(
        self,
        hyperparameters: Dict,
    ) -> None:
        self.hyperparameters = hyperparameters

    @property
    def logger(self) -> WandbLogger:
        return WandbLogger(
            project=self.hyperparameters.project_name,
            name=self.hyperparameters.model_name,
        )


class WandbLoggerV2:
    def __init__(
        self,
        hyperparameters: Dict,
        notes: str = "",
        tags: List[str] = [],
    ) -> None:
        self.hyperparameters = hyperparameters
        self.notes = notes
        self.tags = tags

    @property
    def logger(self) -> WandbLogger:
        return WandbLogger(
            project=self.hyperparameters.project_name,
            name=self.hyperparameters.model_name,
            notes=self.notes,
            tags=self.tags,
        )
