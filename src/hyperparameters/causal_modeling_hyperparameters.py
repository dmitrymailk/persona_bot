from dataclasses import dataclass


@dataclass
class BaseHyperparametersV1:
    """
    pad_token_id - id который стоит игнорировать при обучении
    """

    weight_decay: float = 0.0
    train_batch_size: int = 16
    valid_batch_size: int = 16
    pad_token_id: int = -100
    learning_rate: float = 1e-4
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    debug_status: int = 0


@dataclass
class PersonaChatHyperparametersV1(BaseHyperparametersV1):
    """
    chat_history_pair_length: int - количество пар диалога с конца
    max_tokens_length: int - максимальная длина последовательности
    """

    train_batch_size: int = 16
    valid_batch_size: int = 16
    chat_history_pair_length: int = 1
    max_tokens_length: int = 512
    model_name: str = "gpt2"
    model_architecture: str = "causal"
    max_response_length: int = 128
    project_name: str = "persona_bot"
    predicted_texts_folder: str = (
        "/home/dimweb/Desktop/deeppavlov/persona_bot/predicted_texts"
    )
    debug_status: int = 0
