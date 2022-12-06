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
<bos> <persona> persona_fact[0]<p_sep>persona_fact[1]<p_sep>persona_fact[2]<p_sep>persona_fact[3]<p_sep>persona_fact[4]<p_sep> <chat> реплика[-6]<с_sep>реплика[-5]<с_sep>реплика[-4]<с_sep>реплика[-3]<с_sep>реплика[-2]<response>
таргет: реплика[-1] <eos>

CausalLM:
входная последовательность:
<bos> <persona> persona_fact[0]<p_sep>persona_fact[1]<p_sep>persona_fact[2]<p_sep>persona_fact[3]<p_sep>persona_fact[4]<p_sep> <chat> реплика[-6]<с_sep>реплика[-5]<с_sep>реплика[-4]<с_sep>реплика[-3]<с_sep>реплика[-2]<response>реплика[-1]<eos_token>
таргет: входная последовательность сдвинутая на 1 вправо

<с_sep> - специальный токен, который разделяет реплики.
<p_sep> - специальный токен, который разделяет персону.
<chat> - специальный токен, который разделяет реплики от персоны.
<persona> - специальный токен, который разделяет персону от реплик.
<response> - специальный токен, который разделяет реплики от ответа.

```

#### hypothesis 3
```text
попробовать случайно перемешать порядок предложений в персоне. в остальном все остальное также как и в hypothesis 2
```

#### hypothesis 4
```text
Seq2Seq:
входная последовательность:
<bos> <persona> persona_fact[0]persona_fact[1]persona_fact[2]persona_fact[3]persona_fact[4]<sep>реплика[-6] реплика[-5] ... <query>реплика[-2]<query/><eos>
таргет:<bos><response>реплика[-1]<response/><eos>

<sep> - специальный токен, раздедяющий токен
<query> - специальный токен, который оборачивает последнюю реплику пользователя
<query/> - 
<response> - специальный токен, оборачивает ответ пользователя
<response/> 
```


#### hypothesis 5
```text
тоже самое что и в hypothesis 4, но теперь исполььзую датасет FoCus	
```

#### hypothesis 6
```text
теперь мы берем датасет ru persona chat.
остальной набор остается неизменным как в hypothesis 3.

разве что меняем заменяем оригинальные модели на мультиязычные, так как
в них более оптимизированные токенизаторы и с ними получается меньше токенов, чем у исходных.
```

- [package project](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [install project from git](https://stackoverflow.com/questions/15268953/how-to-install-python-package-from-github)

```bash
python3 -m build
```

```bash
twine upload dist/*
```
