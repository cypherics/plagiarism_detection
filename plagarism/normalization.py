import string, re, nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

pattern_at_user = r"(@)user+|@"
pattern_digits = r"\d+(nd|th|st)*"
pattern_space = r"\s{2,}"
pattern_special_chars = r"[^\w\s]|(_)+"
pattern_url = r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b"
pattern_single_letter = r"\b[a-z]{1}\b"

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


def case_conversion(text: string):
    return text.lower()


def apply_regex(text: string):
    text = re.sub(pattern_url, "", text)
    text = re.sub(pattern_at_user, "", text)
    text = re.sub(pattern_digits, "", text)
    text = re.sub(pattern_special_chars, " ", text)
    text = re.sub(pattern_single_letter, "", text)
    text = re.sub(pattern_space, " ", text)
    return text


def remove_stop_words(text: string):
    stop_words = set(stopwords.words("english"))
    tokenized_text = word_tokenize(text)
    words_filtered = []

    for w in tokenized_text:
        if w not in stop_words:
            words_filtered.append(w)
    return words_filtered


def lemmatization(text: string):
    word_lemma = WordNetLemmatizer()
    _words = []
    for word in text:
        _words.append(word_lemma.lemmatize(word))
    return _words


def text_normalization(text):
    text = case_conversion(text)
    text = apply_regex(text)
    tokenized_text = remove_stop_words(text)
    tokenized_text = lemmatization(tokenized_text)
    return tokenized_text
