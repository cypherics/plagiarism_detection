# Plagarism Detection 
Code for detecting extrinsic and intrinsic plagiarism

### Data format for Extrinsic
.....
### Data format for Intrinsic
.....
### Config
```yaml
extrinsic:
  source:
    # dir where .txt files are stored for source
    dir:
      - dataset/subset/subset_1/sou_1
      - dataset/subset/subset_2/sou_2
      - dataset/subset/subset_3/sou_3
      
    # If pth is not present it will compute do the pre-proceissing 
    # and save them so for next run its will skip the processing and used data from csv
    pth: dataset/source_sent_all_three_subset.csv

  suspicious:
    # dir where .txt files are stored for source
    dir:
      - dataset/subset/subset_1/sus_1
    
    # If pth is not present it will compute do the pre-proceissing 
    # and save them so for next run its will skip the processing and used data from csv
    pth: dataset/suspicious_sent.csv

  index: dataset/output/se_index_subset_1_2_3.index
  save: dataset/output/set1/SE/se_output_subset_1_with_all_three_source.csv

intrinsic:
  suspicious:
    # dir where .txt files are stored for source
    dir:
      - dataset/pan-plagiarism-corpus-2009.part3/pan-plagiarism-corpus-2009/intrinsic-analysis-corpus/suspicious-documents
      - dataset/pan-plagiarism-corpus-2009.part2/pan-plagiarism-corpus-2009/intrinsic-analysis-corpus/suspicious-documents
      - dataset/pan-plagiarism-corpus-2009/intrinsic-analysis-corpus/suspicious-documents
    
    # If pth is not present it will compute do the pre-proceissing 
    # and save them so for next run its will skip the processing and used data from csv
    pth: path/to/suspicious_sent_intrinsic.csv

  save: path/to/save/intrinsic_output.csv

  features:
    - automated_readability_index
    - average_sentence_length_chars
    - average_sentence_length_words
    - average_syllables_per_word
    - average_word_frequency_class
    - average_word_length
    - coleman_liau_index
    - flesch_reading_ease
    - functionword_frequency
    - linsear_write_formula
    - most_common_words_without_stopwords
    - number_frequency
    - punctuation_frequency
    - sentence_length_distribution
    - special_character_frequency
    - stopword_ratio
    - top_3_gram_frequency
    - top_bigram_frequency
    - top_word_bigram_frequency
    - uppercase_frequency
    - word_length_distribution
    - yule_k_metric

evaluation:
  results: path/where/results.csv
  ground_truth: path/where/ground_truth.csv
```
### Run Extrinsic
```python
# USING TFIDF FOR FEATURES
python extrinsic_tfidf --config path/to/config.yaml

# USING DISTILL_BERT FOR FEATURES
python extrinsic_se --config path/to/config.yaml
```

### Run Intrinsic
```python
python intrinsic --config path/to/config.yaml
```