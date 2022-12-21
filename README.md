### Dataset
- [original_dataset](https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json)

#### hypothesis 1
```text
использовал CausalLM и Seq2Seq. Seq2Seq показал себя лучше.
Seq2Seq:
входная последовательность: сконкатенированная персона + история диалога(кроме последнего ответа)
таргет: ответ от пользователя
CausalLM:
входная последовательность: сконкатенированная персона + вся история диалога(некоторое количество пар)
таргет: входная последовательность сдвинутая на 1 вправо

обрезаю персону по длине согласно 0.95 квантилю длины персоны в датасете по токенам. тоже самое и с репликами
```

#### hypothesis 1.1

```text
CausalLM:
	тоже самое как hypothesis 1 только перемешиваю персону при трейне.
```

#### hypothesis 1.2

```text
CausalLM:
	тоже самое как hypothesis 1 только перемешиваю персону и диалог при трейне.
(результат как и ожидалось очень низкий)
```

### hypothesis 2

- [Позаимствовано у GOBEL.](https://huggingface.co/spaces/microsoft/GODEL-Demo/blob/main/app.py#L61)
```text
Seq2Seq:
входная последовательность: [CONTEXT] dialog_1 EOS dialog_2 EOS ... dialog_n-1 [KNOWLEDGE] persona_1 persona_2 ... persona_n  
таргет: ответ от пользователя
```

#### hypothesis 2.1

```text
как hypothesis 2, только перемешивать персону при трейне
```

#### hypothesis 3
- русский бот
```text
исходный русский ru persona chat.
все остальное как в hypothesis 2.1
```

#### hypothesis 4
- русский бот
```text
все как в hypothesis 3. только теперь брать обе персоны и менять диалоги местами.

так как у нас есть 2 персоны, то мы можем использовать персону при трейне с 2 сторон. С другой стороны может возникнуть утечка данных, то есть когда даже я не буду прописывать вторую персону для ответа, модель как бы будет о ней знать, потому что она раньше видела ее, но с другой стороны.
```

# Russian bot
- [ru_persona_chat]
- [cornell_movie_corpus]

- [package project](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [install project from git](https://stackoverflow.com/questions/15268953/how-to-install-python-package-from-github)

```bash
python3 -m build
```

```bash
twine upload dist/*
```

```bash
du -sh -- *
```
