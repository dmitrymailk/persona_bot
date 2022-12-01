import peewee as pw
import pandas as pd
from dimweb_persona_bot.database_logger.models import ModelPredictionV1, ModelMetricsV1
from dimweb_persona_bot.database_logger.logger import DatabaseAnalaizerV1
from typing import List


class DatabaseAnalyzerV1:
    def __init__(
        self,
        host: str,
        port: int,
    ):
        self.db = DatabaseAnalaizerV1(
            host=host,
            port=port,
        )
        self.get_random_row = """
        WITH random_row AS (
            SELECT * FROM model_prediction_v1 ORDER BY random() LIMIT 1
        )
        SELECT * FROM random_row;
        """

    def get_random_sample_by_wandb_run_id(
        self,
        wandb_run_id: str,
        table_save_path: str = "./temp.csv",
        columns: List[str] = [
            "wandb_run_id",
            "model_name",
            "epoch",
            "context",
            "model_prediction",
            "actual_response",
        ],
    ):
        random_row = self.db.execute_sql(self.get_random_row)
        random_row = random_row[0][1]

        data_1 = (
            ModelPredictionV1.select()
            .where(
                (ModelPredictionV1.prediction_id == random_row)
                & (ModelPredictionV1.wandb_run_id == wandb_run_id)
            )
            .order_by(ModelPredictionV1.id)
            .dicts()
        )

        data_2 = pd.DataFrame.from_records(data_1)
        data_2[columns].to_csv(table_save_path, index=False)

    def get_sample_with_metrics_by_wandb_run_id(
        self,
        wandb_run_id: str,
        table_save_path: str = "./temp.csv",
        columns: List[str] = [
            "wandb_run_id",
            "model_name",
            "prediction_id",
            "epoch",
            "context",
            "model_prediction",
            "actual_response",
            "valid_loss_epoch",
            "blue_score_epoch",
            "rougel_score_epoch",
            "charf_score_epoch",
        ],
        samples_amount: int = 1,
    ):
        """
        similar to
        select * from public.model_prediction_v1 mp
        join model_metrics_v1 mmv on mp.epoch = mmv.epoch
        where
        mp.prediction_id = '69_2' and
        mp.wandb_run_id = '19rmj0yc' and
        mmv.wandb_run_id = '19rmj0yc'
        """
        data = []
        for _ in range(samples_amount):
            random_row = self.db.execute_sql(self.get_random_row)
            random_row = random_row[0][1]

            data_1 = (
                ModelPredictionV1.select(ModelPredictionV1, ModelMetricsV1)
                .join(
                    ModelMetricsV1,
                    pw.JOIN.INNER,
                    on=(ModelMetricsV1.epoch == ModelPredictionV1.epoch),
                )
                .where(
                    (ModelMetricsV1.wandb_run_id == wandb_run_id)
                    & (ModelPredictionV1.wandb_run_id == wandb_run_id)
                    & (ModelPredictionV1.prediction_id == random_row)
                )
                .order_by(ModelPredictionV1.id)
                .dicts()
            )

            data_2 = pd.DataFrame.from_records(data_1)
            data.append(data_2)

        data = pd.concat(data)
        data[columns].to_csv(table_save_path, index=False)
        return data[columns]

    def get_sample_with_metrics_by_wandb_run_id_and_sample_id(
        self,
        wandb_run_ids: list,
        sample_ids: list,
        table_save_path: str = "./temp.csv",
        columns: List[str] = [
            "wandb_run_id",
            "model_name",
            "prediction_id",
            "epoch",
            "context",
            "model_prediction",
            "actual_response",
            "valid_loss_epoch",
            "blue_score_epoch",
            "rougel_score_epoch",
            "charf_score_epoch",
        ],
    ):
        data = []
        for sample_id in sample_ids:
            for wandb_run_id in wandb_run_ids:
                data_1 = (
                    ModelPredictionV1.select(ModelPredictionV1, ModelMetricsV1)
                    .join(
                        ModelMetricsV1,
                        pw.JOIN.INNER,
                        on=(ModelMetricsV1.epoch == ModelPredictionV1.epoch),
                    )
                    .where(
                        (ModelMetricsV1.wandb_run_id == wandb_run_id)
                        & (ModelPredictionV1.wandb_run_id == wandb_run_id)
                        & (ModelPredictionV1.prediction_id == sample_id)
                    )
                    .order_by(ModelPredictionV1.id)
                    .dicts()
                )

                data_2 = pd.DataFrame.from_records(data_1)
                data.append(data_2)

        data = pd.concat(data)
        data[columns].to_csv(table_save_path, index=False)
        return data[columns]
