import logging

import nltk

logging.basicConfig(
    format="[%(levelname)s][%(asctime)s]:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("universal_tagset")
