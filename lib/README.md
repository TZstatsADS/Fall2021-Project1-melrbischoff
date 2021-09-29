# ADS Project 1:  R Notebook on the history of philosophy

Required Packages:
    pandas
    re
    gensim
    nltk
    vaderSentiment
    textblob
    wikipedia

This folder contains one notebook `utils_class.ipynb`. In this file is one class that does all of the computation for the project. Below is a description for the class and its functions.

`read_data(self)` reads in the dataset in the relative path `../data/{dataset_name}.csv`

`rem_sw(self, var_in)` removes stopwords from `var_in`. Uses stopwords from the `nltk.stopwords` package
    
`clean_text(self, var_in)` removes special characters from `var_in`
    
`stem_fun(self, var)` stems words in `var` from `nltk.stem` using the `PorterStemmer` function

`add_clean_cols_to_data(self)` adds following colums to `self.data`: 
    `sentence_lowered`: the lower case text of the `sentence_str` col
    `clean_text`: special characters removed from `sentence_lowered` col using `clean_text` func
    `rem_sw`: stopwords removed from `clean_text` col using `rem_sw` func
    `rem_sw_stem`: stems words from `rem_sw` col using `stem_fun` func

`get_author_wikipedia_page(self)` leverages the `wikipedia` package to get wikipedia biography on each author in `self.data`. Saves object `self.author_wiki_dict`. Also prints list of authors who it is unable to find a wikipedia biography page for.

`get_school_wikipedia_page(self)` does the same as `get_author_wikipedia_page()` but for each school in `self.data`. Saves object `self.school_wiki_dict`.

`get_author_sexes(self)` uses pronouns 'she','her','hers' for females and 'he','him','his' for males to determine if the author is a male or female based on their wikipedia biography. Saves object `self.sex_dict`.    

`lda_fun(self, df_in, n_topics_in, num_words_in)` gets the number of topics, `n_topics_in`, containing `num_words_in` words for a dataset, `df_in`. Uses LDA to determine topics. 

`run_and_write_lda_authors_fun(self)` runs the `lda_fun` function on `self.data` for each author. Saves object `self.lda_author_topics_df` and writes it to `../output/lda_author_topics_df.csv'`

`run_and_write_lda_school_fun(self)` runs the `lda_fun` function on `self.data` for each school. Saves object `self.lda_school_topics_df` and writes it to `../output/lda_school_topics_df.csv'`


def get_sentiment_words(self):
        # TO DO - change to rel path
        file_names = ['positive-words', 'negative-words']
        pos_neg_dict = {}
        for file in file_names:
            path = "/Users/melissa/Desktop/columbia/class/Applied_Data_Science/Fall2021-Project1-melrbischoff/data/{}.txt".format(
                file
            )
            with open(path, "r", encoding="ISO-8859-1") as f:
                contents = []
                for line in f:
                    line = line.strip()
                    contents.append(line)
            f.close()
            pos_neg_dict[file] = contents

    def gen_senti(self, arbitrary_text):
        '''
        Tokenizes arbitrary text and compares each token with the positive and 
        negative lexicons of each dictionary and outputs the sentiment score, S
        '''
        import re
        arbitrary_text_clean = re.sub(r'[^a-zA-Z ]+', '', arbitrary_text)
        arbitrary_text_list = arbitrary_text_clean.split()

        pw = [-1 for word in arbitrary_text_list if word in (pos_neg_dict['negative-words'])]
        nw = [1 for word in arbitrary_text_list if word in (pos_neg_dict['positive-words'])]
        pc = len(pw)
        nc = len(nw)
        total = pc + nc
        try:
            S = (sum(pw) + sum(nw)) / total
        except ZeroDivisionError:
            S = None
        return S
    
    def vader_senti(self):
        vaderSent = SentimentIntensityAnalyzer()
    
    def gen_textblob_senti(self, var_in):
        blob = TextBlob(var_in)
        return blob.sentiment.polarity
    
    def run_sentiment_analysis(self):
        self.data['simple_senti'] = self.data.rem_sw_stem.apply(gen_senti)
        self.data['vader'] = self.data.rem_sw_stem.apply(
            lambda x: vaderSent.polarity_scores(x)['compound']
        )
        self.data['textblob_senti'] = self.data.rem_sw_stem.apply(gen_textblob_senti)