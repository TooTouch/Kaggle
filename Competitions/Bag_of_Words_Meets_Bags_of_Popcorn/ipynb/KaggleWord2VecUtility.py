import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from multiprocessing import Pool

class KaggleWord2VecUtility(object):

    def review_to_wordlist(self, review, remove_stopwords=False):
        # 1. HTML 제거
        review_text = BeautifulSoup(review, "html.parser").get_text()
        # 2. 특수문자를 공백으로 바꿔줌
        review_text = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. 소문자로 변환 후 나눈다.
        words = review_text.lower().split()
        # 4. 불용어 제거
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        # 5. 어간추출
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
        # 6. 리스트 형태로 반환
        return(words)


    def review_to_join_words(self,  review, remove_stopwords=False ):
        words = self.review_to_wordlist(review, remove_stopwords=False)
        join_words = ' '.join(words)
        return join_words


    def review_to_sentences(self,  review, remove_stopwords=False ):
        # punkt tokenizer를 로드한다.
        """
        이 때, pickle을 사용하는데
        pickle을 통해 값을 저장하면 원래 변수에 연결 된 참조값 역시 저장된다.
        저장된 pickle을 다시 읽으면 변수에 연결되었던
        모든 레퍼런스가 계속 참조 상태를 유지한다.
        """
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # 1. nltk tokenizer를 사용해서 단어로 토큰화 하고 공백 등을 제거한다.
        raw_sentences = tokenizer.tokenize(review.strip())
        # 2. 각 문장을 순회한다.
        sentences = []
        for raw_sentence in raw_sentences:
            # 비어있다면 skip
            if len(raw_sentence) > 0:
                # 태그제거, 알파벳문자가 아닌 것은 공백으로 치환, 불용어제거
                sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence, remove_stopwords))
        return sentences


    # 참고 : https://gist.github.com/yong27/7869662
    # http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    # 속도 개선을 위해 멀티 스레드로 작업하도록

    def _apply_df(self, args):
        df, func, kwargs = args
        return df.apply(func, **kwargs)


    def apply_by_multiprocessing(self, df, func, **kwargs):
        # 키워드 항목 중 workers 파라메터를 꺼냄
        workers = kwargs.pop('workers')
        # 위에서 가져온 workers 수로 프로세스 풀을 정의
        pool = Pool(processes=workers)
        # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
        result = pool.map(self._apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
        pool.close()
        # 작업 결과를 합쳐서 반환
        return pd.concat(result)


    def getCleanReviews(self, reviews, func, workers, remove_stopwords=False):
        clean_reviews = []
        clean_reviews = self.apply_by_multiprocessing(reviews["review"],
                                                     func,
                                                     workers=workers,
                                                     remove_stopwords=remove_stopwords)
        return clean_reviews

    # Part 2
    def makeFeatureVec(self, words, model, num_features):
        """
        주어진 문장에서 단어 벡터의 평균을 구하는 함수
        """
        # 속도를 위해 0으로 채운 배열로 초기화한다.
        featureVec = np.zeros((num_features,), dtype="float32")

        nwords = 0.
        # Index2word는 모델의 사전에 있는 단어 명을 담은 리스트이다.
        # 속도를 위해 set 형태로 초기화한다.
        index2word_set = set(model.wv.index2word)
        # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model[word])
        # 결과를 단어 수로 나누어 평균을 구한다.
        featureVec = np.divide(featureVec, nwords)
        return featureVec

    def getAvgFeatureVecs(self, reviews, model, num_features):
        # 리뷰 단어 목록의 각각에 대한 평균 feature 벡터를 계산하고
        # 2D numpy 배열을 반환한다.

        # 카운터를 초기화한다.
        counter = 0.
        # 속도를 위해 2D 넘파이 배열을 미리 할당한다.
        reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

        for review in reviews:
            # 매 1000개 리뷰마다 상태를 출력
            if counter % 1000. == 0.:
                print("Review %d of %d" % (counter, len(reviews)))
            # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.
            reviewFeatureVecs[int(counter)] = self.makeFeatureVec(review, model, num_features)
            # 카운터를 증가시킨다.
            counter = counter + 1.
        return reviewFeatureVecs

    # Part 3
    def create_bag_of_centroids(self, wordlist, word_centroid_map):

        # The number of clusters is equal to the highest cluster index
        # in the word / centroid map
        num_centroids = max(word_centroid_map.values()) + 1

        # Pre-allocate the bag of centroids vector (for speed)
        bag_of_centroids = np.zeros(num_centroids, dtype="float32")

        # Loop over the words in the review. If the word is in the vocabulary,
        # find which cluster it belongs to, and increment that cluster count
        # by one
        for word in wordlist:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_of_centroids[index] += 1

        # Return the "bag of centroids"
        return bag_of_centroids