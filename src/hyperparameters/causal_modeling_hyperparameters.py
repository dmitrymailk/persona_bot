from dataclasses import dataclass


@dataclass
class BaseHyperparametersV1:
    """
    pad_token_id - id который стоит игнорировать при обучении
    """

    train_batch_size: int = 16
    valid_batch_size: int = 16
    pad_token_id: int = -100


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
