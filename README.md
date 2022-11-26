### Dataset
- [original_dataset](https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json)

#### hypothesis 1
```text
использовал CausalLM и Seq2Seq. Seq2Seq показал себя лучше.
Seq2Seq:
входная последовательность: сконкатенированная персона + chat: + последняя реплика от пользователя
таргет: ответ от пользователя
CausalLM:
входная последовательность: сконкатенированная персона + последняя реплика от пользователя+ответ от пользователя
таргет: входная последовательность сдвинутая на 1 вправо
```

#### hypothesis 2
```text
Seq2Seq:
входная последовательность:
<persona> persona_fact[0]<p_sep>persona_fact[1]<p_sep>persona_fact[2]<p_sep>persona_fact[3]<p_sep>persona_fact[4]<p_sep> <chat> реплика[-6]<с_sep>реплика[-5]<с_sep>реплика[-4]<с_sep>реплика[-3]<с_sep>реплика[-2]<response>
таргет: реплика[-1]

CausalLM:
входная последовательность:
<persona> persona_fact[0]<p_sep>persona_fact[1]<p_sep>persona_fact[2]<p_sep>persona_fact[3]<p_sep>persona_fact[4]<p_sep> <chat> реплика[-6]<с_sep>реплика[-5]<с_sep>реплика[-4]<с_sep>реплика[-3]<с_sep>реплика[-2]<response>реплика[-1]<eos_token>
таргет: входная последовательность сдвинутая на 1 вправо

<с_sep> - специальный токен, который разделяет реплики.
<p_sep> - специальный токен, который разделяет персону.
<chat> - специальный токен, который разделяет реплики от персоны.
<persona> - специальный токен, который разделяет персону от реплик.
<response> - специальный токен, который разделяет реплики от ответа.

```