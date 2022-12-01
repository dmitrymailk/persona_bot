## Postgresql
### выбрать случайный элемент из списка и потом записать его в переменную
```sql
WITH random_row AS (
	SELECT * FROM model_prediction_v1 ORDER BY random() LIMIT 1
)
SELECT * FROM random_row;
```
