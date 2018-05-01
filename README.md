# GSDTM and LMDTM

A TensorFlow implementation of the GSDTM and LMDTM models, published in IJCNN 2018 conference with the name "Topic Modeling using Variational Auto-Encoders with Gumbel-Softmax and Logistic-Normal Mixture Distributions". 

## Requirements

This implementation requires Python 3 and Tensorflow (above version 1.2). All the code was tested using Python `3.6.5` and Tensorflow `1.4.1`. All the dependencies can be installed via pip:

        $ pip install -r requirements.txt
        $ python -c "import nltk; nltk.download('punkt')"


## Preprocessing for 20newsgroups dataset

In order to preprocess the 20newsgroups dataset, you can use the preprocessing script:
        
        $ python preprocess.py --output <path to preprocessed dataset directory> --vocab <path to vocabulary file>


* `<path to preprocessed dataset directory>` is the folder where the preprocessed data will be stored. 
* `<path to vocabulary file>` is the path for vocabulary file.

## Dataset for RCV1-v2
The RCV1-v2 preprocessed files can be downloaded here: [RCV1-v2 Dataset](https://drive.google.com/drive/folders/1VLrtUfVwUMG9OSycmZTecKCQKLi24Hmr?usp=sharing). 

## Datasets format

The dataset is located on folder `data` and it is divided in three CSV files: `train.feat` for training, `valid.feat` for validation and `test.feat` for testing. Each CSV file contains two string fields separated with a blank delimiter represent a document: the first field is the label, while the second represents the frequencies of each token that appears in the document. Each line of the vocabulary file represents one token (word), identified by the number of line where it is located. The vocabulary file used in the experiments can be found in folder `data`.


## Training

To train the models with default parameters, you must provide the path to preprocessed dataset and the name of model that will be trained ('gsdtm' for GSDTM and 'lmdtm' for LMDTM model):

        $ python training.py --data_dir <path to preprocessed dataset> --model <name of model>

To explore all additional parameters, please use the help option:

        $ python training.py --help


## Evaluating results

To evaluate the results in terms of quality of topics, perplexity and precision by fraction of documents:

        $ python evaluate.py --summaries_dir <path to log directory> 

