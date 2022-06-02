# Plagarism Detection 

### [REFERENCE]

1 - http://ceur-ws.org/Vol-502/paper9.pdf

2 - http://ceur-ws.org/Vol-1177/CLEF2011wn-PAN-OberreuterEt2011.pdf

3 - https://ccc.inaoep.mx/~villasen/bib/Intrinsic%20Plagiarism%20Detection.pdf

4 - https://aclanthology.org/N16-1175.pdf

### [Evaluation]

1 - https://pan.webis.de/sepln09/pan09-web/intrinsic-plagiarism-detection.html

2 - https://pan.webis.de/sepln09/pan09-web/external-plagiarism-detection.html

3 - https://github.com/pan-webis-de/pan-code/tree/master/sepln09

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