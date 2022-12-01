# auto-generated snapshot
from peewee import *
import datetime
import peewee


snapshot = Snapshot()


@snapshot.append
class ModelMetricsV1(peewee.Model):
    epoch = IntegerField()
    model_name = CharField(max_length=255)
    model_architecture = CharField(max_length=255)
    wandb_run_id = CharField(max_length=255)
    valid_loss_epoch = FloatField()
    blue_score_epoch = FloatField()
    rougel_score_epoch = FloatField()
    charf_score_epoch = FloatField()
    debug_status = IntegerField()
    class Meta:
        table_name = "model_metrics_v1"


@snapshot.append
class ModelPredictionV1(peewee.Model):
    prediction_id = CharField(max_length=255)
    epoch = IntegerField()
    model_name = CharField(max_length=255)
    model_architecture = CharField(max_length=255)
    wandb_run_id = CharField(max_length=255)
    persona = TextField()
    context = TextField()
    model_prediction = TextField()
    actual_response = TextField()
    debug_status = IntegerField()
    class Meta:
        table_name = "model_prediction_v1"


