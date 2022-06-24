# Plagarism Detection 

### [REFERENCE]

1 - http://ceur-ws.org/Vol-502/paper9.pdf

2 - http://ceur-ws.org/Vol-1177/CLEF2011wn-PAN-OberreuterEt2011.pdf

3 - https://ccc.inaoep.mx/~villasen/bib/Intrinsic%20Plagiarism%20Detection.pdf

4 - https://aclanthology.org/N16-1175.pdf

5 - http://rochi.utcluj.ro/ijusi/articles/IJUSI-13-1-Rosu.pdf

6 - https://pub.tik.ee.ethz.ch/students/2018-HS/BA-2018-19.pdf

7 - https://www.fruct.org/publications/fruct25/files/Lag.pdf

8 - https://www.researchgate.net/publication/303382784_Using_Stylometric_Features_for_Sentiment_Classification

9 - http://ceur-ws.org/Vol-502/paper8.pdf

10 - https://link.springer.com/content/pdf/10.1007%2Fs10579-010-9115-y.pdf

### [Evaluation]

1 - https://pan.webis.de/sepln09/pan09-web/intrinsic-plagiarism-detection.html

2 - https://pan.webis.de/sepln09/pan09-web/external-plagiarism-detection.html

3 - https://github.com/pan-webis-de/pan-code/tree/master/sepln09


### [Intrinsic]
1 - https://github.com/Coolcumber/inpladesys

2 - https://github.com/mullerpeter/authorstyle/blob/master/authorstyle/features/stylometric_features.py


- ##### Features
    
    ```text
    automated_readability_index
    average_sentence_length_chars
    average_sentence_length_words
    average_syllables_per_word
    average_word_frequency_class
    average_word_length
    coleman_liau_index
    flesch_reading_ease
    functionword_frequency
    linsear_write_formula
    most_common_words_without_stopwords
    number_frequency
    pos_tag_frequency
    pos_tag_trigram_frequency
    punctuation_frequency
    sentence_length_distribution
    sichel_s_metric
    smog_index
    special_character_frequency
    stopword_ratio
    top_3_gram_frequency
    top_bigram_frequency
    top_word_bigram_frequency
    uppercase_frequency
    word_length_distribution
    yule_k_metric
    ```

### [GENERATE GROUND TRUTH FROM XML]
```python
import os

import pandas as pd

so = {}
su = {}
ALL_DIR = [r"C:\Users\Fuzail.Palnak\Downloads\pan-plagiarism-corpus-2009.part2\pan-plagiarism-corpus-2009\external-analysis-corpus",
       r"C:\Users\Fuzail.Palnak\Downloads\pan-plagiarism-corpus-2009.part3\pan-plagiarism-corpus-2009\external-analysis-corpus",
    r"C:\Users\Fuzail.Palnak\Downloads\pan-plagiarism-corpus-2009\external-analysis-corpus"]

gt = pd.DataFrame()

for dir in ALL_DIR:
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.xml'):
                if "suspicious-document" in file:
                    q = pd.read_xml(os.path.join(root, file))
                    q["suspicious_reference"] = file
                    gt = pd.concat([gt, q], ignore_index=True, sort=False)

for dir in ALL_DIR:
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.txt'):
                if "source-document" in file:
                    so[file] = (os.path.join(root, file))
                elif "suspicious-document" in file:
                    su[file] = (os.path.join(root, file))


gt["source_text"] = None
gt["suspicious_text"] = None

suspicious_reference = gt["suspicious_reference"].unique().tolist()

for j in suspicious_reference:
    s = gt.loc[gt['suspicious_reference'] == j]
    pg = s.loc[s['name'] == "artificial-plagiarism"]

    if not pg.empty:
        for i, row in pg.iterrows():
            so_r = so[row["source_reference"]]
            su_r = su[row["suspicious_reference"].replace(".xml", ".txt")]

            with open(so_r, encoding="utf8") as f:
                so_lines = f.read()

            with open(su_r, encoding="utf8") as f:
                su_lines = f.read()

            so_txt = so_lines[int(row["source_offset"]):int(row["source_offset"])+int(row["source_length"])]
            su_txt = su_lines[int(row["this_offset"]):int(row["this_offset"])+int(row["this_length"])]

            gt.loc[i, 'source_text'] = so_txt.replace('\n', ' ')
            gt.loc[i, 'suspicious_text'] = su_txt.replace('\n', ' ')

gt.to_csv("gt.csv", index=False)
```