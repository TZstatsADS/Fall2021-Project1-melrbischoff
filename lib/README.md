# ADS Project 1:  R Notebook on the history of philosophy

Required Packages:
    pandas
    re
    gensim
    nltk
    vaderSentiment
    textblob
    wikipedia

This folder contains one notebook `utils.ipynb`. In this file the code that does all of the computation for the project. Below is a description of the code.

functions:
`read_data()` reads in the dataset in the relative path `../data/{dataset_name}.csv`

`rem_sw(var_in)` removes stopwords from `var_in`. Uses stopwords from the `nltk.stopwords` package
    
`clean_text(var_in)` removes special characters from `var_in`
    
`stem_fun(var)` stems words in `var` from `nltk.stem` using the `PorterStemmer` function

`get_author_wikipedia_page()` leverages the `wikipedia` package to get wikipedia biography on each author in `self.data`. Also prints list of authors who it is unable to find a wikipedia biography page for.

`get_author_sexes()` uses pronouns 'she','her','hers' for females and 'he','him','his' for males to determine if the author is a male or female based on their wikipedia biography.

**Sentiment Analysis**
`VADER` (Valence aware dictionary for sentiment reasoning) is a tool specifically designed for sentiment analysis and is attuned to sentiments expressed in social media. It uses a list of lexical features which are labeled as positive or negative according to their semantic orientation to calculate the text sentiment.  VADER sentiment returns the probability of a given input sentence to be positive, negative, and neutral, which add up to 100%, along with a compound score.

`TextBlob` is a python library that offers a simple API for diving into common natural language processing tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more. TextBlob sentiment analyzer returns two properties for a given input sentence: Polarity is a float which lies in the range of [-1,1], which is like VADER’s compound score; Subjectivity is a float in the range of [0,1]. Subjective sentences, which are marked 1, refer to personal opinion, emotion or judgment, while objective ones refer to fact-based information. Textblob disregards the words that it has no acquaintance with and only considers expressions that it can dole out extremity and midpoints to get the last score.

Both `VADER` and `textblob` are rule-based. That means we can perform modeling on raw text with minimal pre-work. The main drawback with the rule-based approach for sentiment analysis is that the method only cares about individual words and completely ignores the context in which it is used. For example, “the party was savage” will be negative when considered by any token-based algorithms.

Although both libraries output relatively similar results, `VADER` picks up more of the positive and negative tones from slang, emojis, etc, which TextBlob missed out on. — whereas TextBlob performs strongly with more formal language usage.

`Gen_senti` is a uses a positive and negative word list (Hu and Bing 2004) `../data/negative-words.txt` and `../data/positive-words.txt`. The function Tokenizes text and compares each token with the positive and negative lexicons of each dictionary and outputs the sentiment score. Positive and negative words count as a score of 1 and -1 respectively for each word matched. The total count for pw and nw are pc and nc, respectively. Each message sentiment is normalized between -1 and 1.