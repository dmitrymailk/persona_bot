from typing import TypedDict, List

from src.hyperparameters.causal_modeling_hyperparameters import (
    H1PersonaChatHyperparametersV1,
)
from src.dataloaders.persona_chat_dataloaders import PersonaChatDatasetSampleV1
from src.utils import flat_list
from src.dataloaders.causal_samplers.causal_samplers_hypothesis_2 import (
    H2CausalTrainPersonaSampleV1,
)
from src.dataloaders.seq2seq_samplers.seq2seq_samplers_hypothesis_1 import (
    H1Seq2SeqSampleDictV1,
    H1Seq2SeqSampleDictV2,
)

from transformers import AutoTokenizer


class H2Seq2SeqTrainPersonaSampleV1(H2CausalTrainPersonaSampleV1):
    """
    seq2seq hypothesis_2
    input_ids:
        bos_token <persona> persona_fact[0]<p_sep> ... persona_fact[4] <chat> реплика[-6]<с_sep> ... реплика[-4]<response> eos_token
    labels:
        bos_token реплика[-1] eos_token
    """

    def _add_sep_token_chat_train(self, input_ids: List[List[int]]) -> List[int]:
        result = []
        for pos, item in enumerate(input_ids):
            result.extend(item)
            if pos < len(input_ids) - 1:
                result.append(self.c_sep_id)
            if pos == len(input_ids) - 1:
                result.append(self.response_id)

        return result

    def get_sample(self) -> H1Seq2SeqSampleDictV1:
        history = self.dataset_sample["history"]
        history = history[-self.hyperparameters.chat_history_pair_length * 2 :]
        labels = history.pop()
        persona = self.dataset_sample["persona"]

        encoded_persona = self.tokenizer.batch_encode_plus(
            persona,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.persona_max_length,
        )

        encoded_persona = self._add_sep_token_persona(
            input_ids=encoded_persona["input_ids"],
        )

        encoded_history = self.tokenizer.batch_encode_plus(
            history,
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )
        encoded_history = self._add_sep_token_chat_train(
            input_ids=encoded_history["input_ids"],
        )

        encoded_labels = self.tokenizer.batch_encode_plus(
            [labels],
            add_special_tokens=False,
            truncation=True,
            max_length=self.hyperparameters.chat_max_length,
        )

        encoded_labels = flat_list(encoded_labels["input_ids"])

        bos_token = self.get_bos_token_id()

        input_ids = [
            *bos_token,
            self.persona_id,
            *encoded_persona,
            self.chat_id,
            *encoded_history,
            self.tokenizer.eos_token_id,
        ]

        attention_mask = [1] * len(input_ids)

        labels = [
            *bos_token,
            *encoded_labels,
            self.tokenizer.eos_token_id,
        ]

        return H1Seq2SeqSampleDictV1(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def get_bos_token_id(self) -> list:
        bos_token = []
        if not "t5" in self.hyperparameters.model_name:
            if self.tokenizer.bos_token is not None:
                bos_token = [self.tokenizer.bos_token_id]

        return bos_token


class H2Seq2SeqValidPersonaSampleV1(H2Seq2SeqTrainPersonaSampleV1):
    """
    input_ids: all persona + history + eos
    labels: user response + eos
    """

    def get_sample(self) -> H1Seq2SeqSampleDictV2:
        persona = self.dataset_sample["persona"]
        sample_id = self.dataset_sample["sample_id"]

        train_sample = super().get_sample()
        [
            "i like to remodel homes .",
            "i like to go hunting .",
            "i like to shoot a bow .",
            "my favorite holiday is halloween .",
        ]
        [
            "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .",
            "you must be very fast . hunting is one of my favorite hobbies .",
            "i am ! for my hobby i like to do canning or some whittling .",
            "i also remodel homes when i am not out bow hunting .",
            "that's neat . when i was in high school i placed 6th in 100m dash !",
            "that's awesome . do you have a favorite season or time of year ?",
            "i do not . but i do have a favorite meat since that is all i eat exclusively .",
            "what is your favorite meat to eat ?",
            "i would have to say its prime rib . do you have any favorite foods ?",
            "i like chicken or macaroni and cheese .",
            "do you have anything planned for today ? i think i am going to do some canning .",
            "i am going to watch football . what are you canning ?",
            "i think i will can some jam . do you also play footfall for fun ?",
        ]
        "<s><persona>i like to remodel homes.<p_sep>i like to go hunting.<p_sep>i like to shoot a bow.<p_sep>my favorite holiday is halloween.<chat>hi, how are you doing? i'm getting ready to do some cheetah chasing to<c_sep>you must be very fast. hunting is one of my favorite hobbies.<responce>i am! for my hobby i like to do canning or some whittling.</s>"

        return H1Seq2SeqSampleDictV2(
            input_ids=train_sample["input_ids"],
            labels=train_sample["labels"],
            custom_labels=train_sample["labels"],
            attention_mask=train_sample["attention_mask"],
            sample_id=sample_id,
            persona=persona,
        )
