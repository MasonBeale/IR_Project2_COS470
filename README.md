# IR_Project2_COS470
This project implements both a Bi-Encoder (using `SentenceTransformer`) and a Cross-Encoder for re-ranking top results from the Bi-Encoder, applied to information retrieval tasks. The code uses topic and answer files in JSON format, and outputs ranked results in TSV format. The project also trains and fine tunes models using the qrel file.

## Contributions
- we started by using Abbas' assignment 3 files as a baseline. 
- mason cleaned those files by removing imports and organizing the files. 
- mason trained cross encoder.
- Abbas trained the biencoder.
- mason made results files with base bi-encoder and cross-encoder.
- abbas made results files with fine-tuned bi-encoders and cross-encoders.
- mason made evaluations and graphs for the results.
- both worked on the papar.

## Note 
All files were run on either Mason's computer, usm lab computers, or Google Colab. Different configurations and dependancies may apply to your system. 

## Files

- **topics_1.json** - JSON file with topics for the first query set.
- **topics_2.json** - JSON file with topics for the second query set.
- **Answers.json** - JSON file with answers to be used for ranking.
- **qrel_1.tsv** - tsv file with relevance to topic_1 queries and answers

## Models
- [**all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 
- [**cross-encoder/ms-marco-TinyBERT-L-2-v2**](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2) 

## Usage 
make_regular_results.py - returns base Bi-encoder and Cross-Encoder results files for 
topic's 1 and 2.
``` bash
python make_regular_results.py <topics_1.json> <topics_2.json> <Answers.json>
```
make_ft_results.py - returns finetuned Bi-encoder and Cross-Encoder results files for 
topic's 1 and 2.
``` bash
python make_ft_results.py <topics_1.json> <topics_2.json> <Answers.json>
```
train_biencoder.py - trains the Bi-encoder model up to 50 epochs in 16 batches and saves the 10,20,...,50 models and gives results for every model
``` bash
python train_biencoder.py <topics_1.json> <qrel_1.tsv> <Answers.json>
```
train_crossencoder.py - trains the Cross-encoder model up to 2 epochs and saves the model
``` bash
python train_crossencoder.py <topics_1.json> <qrel_1.tsv> <Answers.json>
```

## Requirements

The code requires the following libraries:
- `torch`
- `sentence_transformers`
- `datasets`
- `beautifulsoup4`
- `tqdm`

Run the following command to install the required packages:
```bash
pip install torch sentence-transformers beautifulsoup4 datasets tqdm
```




