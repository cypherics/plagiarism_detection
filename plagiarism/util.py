import string
import re
import nltk
from typing import List, Optional

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords

from plagiarism.constants import PARA_COL, INPUT_COL

pattern_digits = r"\d+(nd|th|st)*"
pattern_space = r"\s{2,}"
pattern_special_chars = r"[^\w\s]|(_)+"
pattern_url = r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b"

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "e.g." in text:
        text = text.replace("e.g.", "e<prd>g<prd>")
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "i.e." in text:
        text = text.replace("i.e.", "i<prd>e<prd>")
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def generate_para_df(filepath):
    para_content = list()
    with open(filepath, "r", encoding="utf-8") as rf:
        _content = []
        for line in rf:
            if line == "\n":
                para_content.append(" ".join(_content))
                _content = []
            else:
                _content.append(line.strip())
    return pd.DataFrame(para_content, columns=[PARA_COL])


def remove_symbols_numbers_letters_consonants(word_token: List):
    clean_token = []
    for token in word_token:
        token = token.lower()
        new_token = re.sub(r"[^a-zA-Z]+", "", token)
        if new_token != "" and len(new_token) >= 2:
            vowels = len([v for v in new_token if v in "aeiou"])
            if vowels != 0:
                clean_token.append(new_token)
    return clean_token


def case_conversion(text: string):
    return text.lower()


def apply_regex(text: string):
    text = re.sub(pattern_url, "", text)
    text = re.sub(pattern_digits, "", text)
    text = re.sub(pattern_special_chars, " ", text)
    text = re.sub(pattern_space, " ", text)
    return text


def remove_stop_words(word_token: List):
    stop_words = set(stopwords.words("english"))
    words_filtered = []

    for w in word_token:
        if w not in stop_words:
            words_filtered.append(w)
    return words_filtered


def sentences_from_para(para):
    return split_into_sentences(para)


def normalize_data(data: str):
    text = case_conversion(data)
    text = apply_regex(text)

    tokenized_text = word_tokenize(text)
    tokenized_text = remove_symbols_numbers_letters_consonants(tokenized_text)
    tokenized_text = remove_stop_words(tokenized_text)
    return tokenized_text


def get_sentences_from_df(data):
    _ip_sent = []
    for idx, row in data.iterrows():
        for sent in sentences_from_para(row[INPUT_COL]):
            _ip_sent.append(sent)
    return _ip_sent
