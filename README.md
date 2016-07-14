# how-question-and-video

##Классы для извлечения признаков
- Класс feature_extractor.HowToQuestionVectorizer извлекать слова, биграмы, специфические окончания запросов.
- Для использования данных о структуре запроса, используется tomita parser. Результат работы парсера сохраняется в xml файл.
Откуда после читается при помощи функции feature_extractor.parse_tomita. Результат работы этой функции можно передавать в
DictVectorizer из модуля sklearn.feature_extraction.

