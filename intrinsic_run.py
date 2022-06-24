from plagiarism.detector import Intrinsic
from plagiarism.doc import SuspiciousDocumentCollection
from plagiarism.vectorizer import StyleEmbedding

suspicious_doc = SuspiciousDocumentCollection(
    pth=r"C:\Users\Fuzail.Palnak\PycharmProjects\crefts\plagiarism_detection\dataset\suspicious_sent.csv",
    dir_iter=["dataset/subset/subset_1/sus_1"],
).extract_sentences()


# f = ["average_sentence_length_words"]
f = [
    "automated_readability_index",
    "average_sentence_length_chars",
    "average_sentence_length_words",
    "average_syllables_per_word",
    "average_word_frequency_class",
    "average_word_length",
    "coleman_liau_index",
    "flesch_reading_ease",
    "functionword_frequency",
    "linsear_write_formula",
    "most_common_words_without_stopwords",
    "number_frequency",
    "punctuation_frequency",
    "sentence_length_distribution",
    "sichel_s_metric",
    "smog_index",
    "special_character_frequency",
    "stopword_ratio",
    "top_3_gram_frequency",
    "top_bigram_frequency",
    "top_word_bigram_frequency",
    "uppercase_frequency",
    "word_length_distribution",
    "yule_k_metric",
]

ii = Intrinsic(suspicious_doc=suspicious_doc, vector_model=StyleEmbedding(f))
ii.query()
