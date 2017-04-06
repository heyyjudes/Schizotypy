#this file will handle the pre-processing after scraping data from Reddit
import os
import re
import csv
from gensim import corpora
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer, sent_tokenize, word_tokenize
import NLPlib
tagger = NLPlib.NLPlib()
contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i had",
"i'll": "i will",
"im" : "i am",
"i'm" : "i am",
"ive" : "i have",
"i've": "i have",
"isn't": "is not",
"it'd": "it had ",
"it'd've": "it would have",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'll": "they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": " where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"you've": "you have"
}

class PreProcessor(object):
    """base class for pre processing"""
    def __init__(self, input_file, dir=False):
        self.texts = []
        self.message_arr = []
        self.author_arr = []
        if dir == True:
            for filename in os.listdir(input_file):
                if filename.endswith('.csv'):
                    self.parse_csv(input_file + "\\" + filename)
                elif filename.endswith('.txt'):
                    self.parse_txt(input_file + "\\" + filename)
                else:
                    print "WARNING: cannot process file type in given directory"
        else:
            if input_file.endswith('.csv'):
                self.parse_csv(input_file)
            elif input_file.endswith('.txt'):
                self.parse_txt(input_file)
            else:
                print "WARNING: no file found check input"

    def parse_csv(self, input_file):
        """function for parsing csv file from reddit scraper"""
        with open(input_file, 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            id = 0
            for row in reader:
                new_message = message.RedMessage(id, row['title'], row['selftext'], row['author_name'], row['subreddit'], row['created_utc'])
                self.message_arr.append(new_message)
                self.texts.append(new_message.body)
                self.texts.append(new_message.title)
                id += 1

    def parse_txt(self, input_file):
        f = open(input_file, 'rb')
        file_arr = []
        messages = f.readlines()
        for m in messages:
            m = m.rstrip('\r\n')
            if len(m) > 0:
                file_arr.append(m)
        self.texts.append(file_arr)

    def filter_message(self):
        #filter by length
        new_arr = []
        for Rmsg in self.message_arr:
            if len(Rmsg.body) > 60:
                new_arr.append(Rmsg)

        self.message_arr = new_arr



class PreProcessorLDA(PreProcessor):
    def __init__(self, input_file, dir=False):
        super(PreProcessorLDA, self).__init__(input_file, dir)
        #cleaned text
        self.tokens = []
        self.sentences = []
        self.sentence_tokens = []
        #removed stop words
        self.no_stop_tokens = []
        #removed stemmed_tokens
        self.stemmed_tokens = []
        self.stemmed_tokens_flattened = []
        self.dictionary = None
        self.corpus = None
        self.tagged_tokens = []

    def tokenize_LDA(self):
        print("tokenizing")
        tokenizer2 = RegexpTokenizer(r'\w+')
        tokenizer1 = WhitespaceTokenizer()
        for i in range(0, len(self.texts)):
            docs = self.texts[i]
            tokens = []
            for doc in docs:
                raw = doc.lower()
                #white space tokenize
                token = tokenizer1.tokenize(raw)
                #extending contractions
                for i in range(0, len(token)):
                    if token[i] in contractions.keys():
                        token[i] = contractions[str(token[i])]
                    #removing links
                    if (re.search('http', token[i])):
                        token[i] = ''

                raw = " ".join(token)
                #regex tokenizing
                token = tokenizer2.tokenize(raw)
                for i in range(0, len(token)):
                    if token[i].isalnum() == False:
                        token[i] = ''
                tokens.append(token)
            self.tokens.append(tokens)
        return self.tokens

    def remove_stop_LDA(self):
        print("removing stops")
        en_stop = get_stop_words('en')
        for i in range(0, len(self.tokens)):
            s_tokens = []
            docs = self.tokens[i]
            for token in docs:
                s_token = [i for i in token if not i in en_stop]
                s_tokens.append(s_token)
            self.no_stop_tokens.append(s_tokens)
        return self.no_stop_tokens


    def stem_LDA(self):
        #create p_stemmer class of Porter Stemmer
        print('Stemming')
        p_stemmer = PorterStemmer()
        for i in range(0, len(self.no_stop_tokens)):
            docs = self.no_stop_tokens[i]
            stemmed_tokens = []
            for s_token in docs:
                text = [p_stemmer.stem(s) for s in s_token]
                #removed to avoid uneven length
                #if text != []:
                stemmed_tokens.append(text)
                self.stemmed_tokens_flattened.append(text)
            self.stemmed_tokens.append(stemmed_tokens)
        return self.stemmed_tokens


    def doc_term_matrix(self):
        #construct document term matrix
        #assign frequencies
        self.dictionary = corpora.Dictionary(self.stemmed_tokens_flattened)
        #converted to bag of words
        self.corpus = [self.dictionary.doc2bow(text) for text in self.stemmed_tokens_flattened]
        #(termID, term frequency)
        return self.corpus, self.dictionary



class PreProcessorFG():

    def __init__(self, control_file, risk_file):
        '''
        patient sentence files
        :param pos_file: extracted control txt file
        :param neg_file: extracted risk txt file
        '''
        cltr = open(control_file)
        risk = open(risk_file)
        self.cltr_texts = cltr.readlines()
        self.cltr_texts = [line.rstrip('\n') for line in self.cltr_texts]
        self.risk_texts = risk.readlines()
        self.risk_texts = [line.rstrip('\n') for line in self.risk_texts]
        self.cltr_sent = []
        self.risk_sent = []
        self.cltr_token = []
        self.risk_token = []
        self.cltr_tagged_token = []
        self.risk_tagged_token = []

    def process(self):
        '''pre-process input files'''
        self.cltr_sent = self.sent_split(self.cltr_texts)
        self.risk_sent = self.sent_split(self.risk_texts)
        self.cltr_sent = self.remove_long_sent(self.cltr_sent)
        self.risk_sent = self.remove_long_sent(self.risk_sent)
        self.cltr_sent = self.sent_preprocess(self.cltr_sent)
        self.risk_sent = self.sent_preprocess(self.risk_sent)

        #tokenizing words
        self.cltr_token = self.word_tokenize(self.cltr_sent)
        self.risk_token = self.word_tokenize(self.risk_sent)

        #tagging words
        self.cltr_tagged_token = self.POS_tag(self.cltr_token)
        self.risk_tagged_token = self.POS_tag(self.risk_token)


    def sent_split(self, input_arr):
        '''
        using nltk to split sentences
        :param input_arr:
        :return:
        '''
        new_sent_arr = []
        for entry in input_arr:
            result = sent_tokenize(entry)
            new_sent_arr += result
        return new_sent_arr

    def remove_long_sent(self, input_arr):
        '''
        Split sentences longer than 30 words
        :param input_arr: array of sentences
        :return:
        '''
        new_sent_arr = []
        for sent in input_arr:
            tokens = word_tokenize(sent)
            if len(tokens) > 30:
                arr = sent.split(',')
                new_arr = []
                for ns in arr:
                    ns_tokens = word_tokenize(ns)
                    if len(ns_tokens) > 5:
                        new_arr.append(ns + ' .')
            else:
                if len(tokens) > 5:
                    new_sent_arr.append(sent)
        return new_sent_arr

    def sent_preprocess(self, input_arr):
        '''
        preprocess sentence: remove all ()
        :param input_arr: array of shortened sentences
        :return: array of sentences with () removed
        '''
        new_arr = []
        for sent in input_arr:
            sent = re.sub('[()]', '', sent)
            new_arr.append(sent)
        return new_arr

    def word_tokenize(self, input_arr):
        '''
        tokenizing word
        :param input_arr: array of preprocessed sentences
        :return: array of tokenized words
        '''
        new_arr = []
        for sent in input_arr:
            new_arr.append(word_tokenize(sent))
        return new_arr

    def POS_tag(self, token_arr):
        '''
        tagging word
        :param token_arr: array of sentence tokens
        :return: array of tagged sentence tokens
        '''
        new_arr = []
        for tokens in token_arr:
            output_tokens = []
            tags = tagger.tag(tokens)
            for i in range(0, len(tokens)):
                new_token = tokens[i] + '/' + tags[i]
                output_tokens.append(new_token)
            new_arr.append(output_tokens)
        return new_arr

    def remove_stop(self, input_arr):
        '''
        return with tokens without stop words
        :param input_arr: tokenized word arrays
        :return: tokenized word arrays with stops removed
        '''
        en_stop = get_stop_words('en')
        s_tokens = []
        for token in input_arr:
            s_token = [i for i in token if not i in en_stop]
            s_tokens.append(s_token)
        return s_tokens

    def stem(self, input_arr):
        '''
        stems tokens in input array
        :param input_arr: tokenized word arrays with stops removed
        :return: array of stemmed words
        '''
        #create p_stemmer class of Porter Stemmer
        p_stemmer = PorterStemmer()
        stemmed_tokens = []
        for s_token in input_arr:
            text = [p_stemmer.stem(s) for s in s_token]
            stemmed_tokens.append(text)
        return stemmed_tokens

