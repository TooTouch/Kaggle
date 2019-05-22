import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stemmer = nltk.stem.PorterStemmer()

def _apply_df(args):
    df, func = args
    return df.apply(func)


def review_to_words(raw_review):
    #1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    #2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    #3. 소문자 변환 후 공백으로 토크나이징
    words = letters_only.lower().split()
    #4. 파이썬은 리스트보다 세트로 찾는게 훨씬 빠름
    stops = set(stopwords.words('english'))
    #5. Stopwords 불용어 제거
    meaningful_words = [w for w in words if not w in stops]
    #6. 어간 추출
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    #7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    return (' '.join(stemming_words))