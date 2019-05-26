import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stemmer = nltk.stem.PorterStemmer()

def _apply_df(args):
    df, func = args
    return df.apply(func)

def make_sentences(reviews):
    sentences = list()
    for review in reviews:
        sentences += review_to_sentences(review)
    return sentences

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
    return ' '.join(stemming_words)


def review_to_wordlist(raw_review, remove_stopwords=False):
    #1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    #2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 소문자 변환 후 공백으로 토크나이징
    meaningful_words = letters_only.lower().split()
    if remove_stopwords:
        #4. 파이썬은 리스트보다 세트로 찾는게 훨씬 빠름
        stops = set(stopwords.words('english'))
        #5. Stopwords 불용어 제거
        meaningful_words = [w for w in words if not w in stops]

    return meaningful_words

# Define a function to split a review into parsed sentences
def review_to_sentences( review, remove_stopwords=False ):
    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # tokenizer를 통해 review를 sentences로 분리한다.
    raw_sentences = tokenizer.tokenize(review.strip())

    # 분리된 리뷰의 문장들을 loop를 통해 wordlist로 변환한다.
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))

    return sentences