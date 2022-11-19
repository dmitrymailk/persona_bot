import peewee as pw


class BaseModel(pw.Model):
    class Meta:
        db = pw.PostgresqlDatabase(
            "nlp_database",
            user="postgres",
            password="postgres",
            host="0.0.0.0",
            port=2345,
            autorollback=True,
        )
        db.connect()

        database = db


class ModelPredictionV1(BaseModel):

    prediction_id = pw.CharField()
    epoch = pw.IntegerField()
    model_name = pw.CharField()
    model_architecture = pw.CharField()
    wandb_run_id = pw.CharField()

    persona = pw.TextField()
    context = pw.TextField()
    model_prediction = pw.TextField()
    actual_response = pw.TextField()

    debug_status = pw.IntegerField()

    class Meta:
        table_name = "model_prediction_v1"


class ModelMetricsV1(BaseModel):

    epoch = pw.IntegerField()
    model_name = pw.CharField()
    model_architecture = pw.CharField()
    wandb_run_id = pw.CharField()

    valid_loss_epoch = pw.FloatField()
    blue_score_epoch = pw.FloatField()
    rougel_score_epoch = pw.FloatField()
    charf_score_epoch = pw.FloatField()

    debug_status = pw.IntegerField()

    class Meta:
        table_name = "model_metrics_v1"
