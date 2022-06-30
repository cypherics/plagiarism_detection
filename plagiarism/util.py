import string
import re
from typing import List

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

INPUT_COL = "para"
PARA_COL = "para"

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


def extrinsic_normalize_data(text: str):
    text = case_conversion(text)
    text = apply_regex(text)

    tokenized_text = word_tokenize(text)
    tokenized_text = remove_symbols_numbers_letters_consonants(tokenized_text)
    tokenized_text = remove_stop_words(tokenized_text)
    return tokenized_text


def intrinsic_normalize_data(text: str):
    text = re.sub(pattern_url, "", text)
    return word_tokenize(text)


def get_sentences_from_df(data):
    _ip_sent = []
    for idx, row in data.iterrows():
        for sent in sentences_from_para(row[INPUT_COL]):
            _ip_sent.append(sent)
    return _ip_sent


def evaluation_iterator(results: str, ground_truth: str):
    gt_df = pd.read_csv(ground_truth)
    output_df = pd.read_csv(results)
    suspicious_reference = gt_df.loc[gt_df.loc[:, "name"] == "artificial-plagiarism"][
        "suspicious_reference"
    ].unique()
    for i, sus in enumerate(tqdm(suspicious_reference)):
        print(f"Loading: {i + 1}/{len(suspicious_reference)}")
        temp_df = output_df.loc[
            output_df.loc[:, "suspicious_filename"] == sus.replace(".xml", ".txt")
        ]
        temp_gt_df = gt_df.loc[gt_df.loc[:, "suspicious_reference"] == sus]
        temp_gt_df = temp_gt_df.loc[
            temp_gt_df.loc[:, "name"] == "artificial-plagiarism"
        ]
        suspicious_text = temp_df["suspicious"].unique()
        suspicious_gt_text = "".join(temp_gt_df["suspicious_text"].to_list())

        yield i, suspicious_text, suspicious_gt_text


def jaccard_similarity(text1: str, text2: str):
    set1 = set(text1)
    set2 = set(text2)
    return float(len(set1.intersection(set2)) / len(set1.union(set2)))


def precision(results: str, ground_truth: str) -> List:
    scores = []
    for i, suspicious_text, suspicious_gt_text in evaluation_iterator(
        results, ground_truth
    ):
        match_len = 0
        for j, suspicious_sentence in enumerate(suspicious_text):
            if suspicious_gt_text.find(suspicious_sentence.strip()) != -1:
                match_len += len(suspicious_sentence.strip())

        scores.append(match_len / len(suspicious_gt_text))

    return scores


def recall(results: str, ground_truth: str) -> List:
    scores = []
    for i, suspicious_text, suspicious_gt_text in evaluation_iterator(
        results, ground_truth
    ):
        tp = 0
        fn = 0
        for j, suspicious_sentence in enumerate(suspicious_text):
            if suspicious_gt_text.find(suspicious_sentence.strip()) != -1:
                tp += 1
            else:
                fn += 1
        if tp + fn == 0:
            if len(suspicious_gt_text) == 0:
                scores.append(1)
            else:
                scores.append(0)
            continue
        scores.append(tp / (tp + fn))
    return scores


def metric(results: str, ground_truth: str):
    return {
        "recall": recall(results, ground_truth),
        "precision": precision(results, ground_truth),
    }
