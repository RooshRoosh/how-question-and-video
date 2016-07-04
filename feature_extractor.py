from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import  DictVectorizer
from scipy.sparse import hstack

import pymorphy2
from pymorphy2.cache import memoized_with_single_argument

m = pymorphy2.MorphAnalyzer()

@memoized_with_single_argument({})
def verb_from(word):
    tags = m.parse(word)[0].tag

    if tags.POS in {'INFN', 'VERB'}:
        r = {
            'number': tags.number,
            'POS': tags.POS,
            'genger': tags.gender,
        }
        return tags.cyr_repr, r
    else:
        return {}


class HowToQuestionVectorizer():

    def __init__(self, **kwargs):

        count_vectorizer_kwargs = dict(ngram_range=(1, 2), min_df=1)
        dict_vectorizer_kwargs = {}

        if 'count_vectorizer' in kwargs:
            count_vectorizer_kwargs = kwargs['count_vectorizer']
        if 'dict_vectorizer' in kwargs:
            dict_vectorizer_kwargs = kwargs['dict_vectorizer']

        self.count_vectorizer = CountVectorizer(**count_vectorizer_kwargs)
        self.additional_vectorizer = DictVectorizer(**dict_vectorizer_kwargs)

    def additional_features(self, raw_documents):

        key_words = [
            'видео', 'фото',
            'бесплатно', 'скачать',
            'на халяву', 'кряк',
        ]

        positions = {
            -1: 'last_word',
            -2: 'word_before_last',
        }

        tokenizer = self.count_vectorizer.build_tokenizer()


        for document in raw_documents:
            sequence = [i.lower() for i in tokenizer(document)]
            d = {}
            d['length'] = len(sequence)
            if d['length'] <2:
                yield d
                continue

            for i in positions:
                if sequence[i] in key_words:
                    d[positions[i]+'='+sequence[i]] = 1

            if sequence[-2] == 'смотреть' and sequence[-1] =='онлайн':
                d['last_word=смотреть_онлайн'] = 1

            d['length'] = len(sequence)
            yield d

    def get_feature_names(self):
        c_features = self.count_vectorizer.get_feature_names()
        extra_features = self.additional_vectorizer.get_feature_names()
        return c_features+extra_features

    def fit_transform(self, raw_documents, y=None):

        c_X = self.count_vectorizer.fit_transform(
            raw_documents, y=None
        )
        extra_X = self.additional_vectorizer.fit_transform(
            self.additional_features(raw_documents)
        )
        return hstack(
            [c_X, extra_X ]
        )

    def transform(self, X, y=None):
        # Добавить обработку неизвестных слов.
        return hstack(
            [
                self.count_vectorizer.transform(X),
                self.additional_vectorizer.transform(self.additional_features(X))
            ]
        )

# vec = DictVectorizer()
# pos_vectorized = vec.fit_transform(pos_window)
# print(pos_vectorized)
# print(pos_vectorized.toarray())
# print(vec.get_feature_names())


# corpus = [
#     'Как чистить грейпфрут фото видео?',
#     'Как чистить потроха рыбы, купленной в магазине Питера смотреть онлайн?',
#     'Можно ли в Питере, после того как ты съел грейпфрут отведать рыбы?',
#     'Какие рыбы водятся в реках Питера?',
# ]

def __example():


    import pymongo
    c = pymongo.MongoClient('localhost', 27017)
    db = c.mailru_qa
    vectorizer = HowToQuestionVectorizer()

    X = vectorizer.fit_transform([item['qtext'] for item in db.question.find({}) if item.get('qtext')])
    print(X.toarray())
    print(vectorizer.get_feature_names())
    print(len(vectorizer.get_feature_names()))
    print(vectorizer.transform(['Какие рыбы водят ся в реках Москвы?']).toarray())


if __name__ == '__main__':
    __example()