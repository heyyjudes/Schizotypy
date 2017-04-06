import nltk
import re

wordlist_url = 'data\\Wordlists\\'
def f_first_person(input_arr):
    keywords = open(wordlist_url + 'First-person').read().splitlines()
    keywords = [z.lower() for z in keywords]
    count = 0

    for token in input_arr:
        token_arr = token.split("/")
        test_token = token_arr[0].lower()
        if test_token in keywords:
            count += 1
    return count

def f_second_person(input_arr):
    keywords = open(wordlist_url + 'Second-person').read().splitlines()
    keywords = [z.lower() for z in keywords]
    count = 0
    for token in input_arr:
        token_arr = token.split("/")
        test_token = token_arr[0].lower()
        if test_token in keywords:
            count += 1
    return count

def f_third_person(input_arr):
    keywords = open(wordlist_url + 'Third-person').read().splitlines()
    keywords = [z.lower() for z in keywords]
    count = 0
    for token in input_arr:
        token_arr = token.split("/")
        test_token = token_arr[0].lower()
        if test_token in keywords:
            print test_token
            count += 1
    return count


def f_conj(input_arr):
    ''' counting coordinating conjunctions CC tags'''
    tag = "CC"
    count = 0
    for token in input_arr:
        token_arr = token.split("/")
        if len(token_arr) > 1:
            if token_arr[1] == tag:
                count += 1
    return count

def f_ing(input_arr):
    ''' counting coordinating conjunctions CC tags'''
    tag = "VBG"
    count = 0
    for token in input_arr:
        token_arr = token.split("/")
        if len(token_arr) > 1:
            if token_arr[1] == tag:
                count += 1
    return count

def f_past_tense_verb(input_arr):
    ''' counting past tense verbs VBD'''
    tag = "VBD"
    count = 0
    for token in input_arr:
        token_arr = token.split("/")
        if len(token_arr) > 1:
            if token_arr[1] == tag:
                count += 1
    return count

def f_verb_future(input_arr):
    ''' counting future tense key words and compositions'''
    tag = "VB"
    keywords = ["'ll", "will", "gonna"]
    count = 0
    sentences = input_arr
    for i in range(0, len(input_arr)):
        token_arr = input_arr[i]
        if len(token_arr) > 1:
            test_token = token_arr[0].lower()
            if test_token in keywords:
                count += 1
            elif i < len(input_arr) - 2:
                #checking for going + to + VB case
                if test_token == "going" and input_arr[i+1].split("/")[0] == "to" and input_arr[i+2].split("/")[1] == tag:
                    count +=1
    return count

def f_exclaim(input_arr):
    ''' counting dashes'''
    tag = "!"
    count = 0
    sentences = input_arr
    for token in input_arr:
        token_arr = token.split("/")
        if len(token_arr) > 1:
            #dashes are not tagged by penn part of speech punctuation
            if token_arr[0] == tag:
                count += 1
    return count

def f_WH_count(input_arr):
    ''' counting wh-words using tags'''
    tags = ["WDT", "WP", "WRB", "WP$"]
    count = 0
    sentences = input_arr
    for token in input_arr:
        token_arr = token.split("/")
        if len(token_arr) > 1:
            if token_arr[1] in tags:
                count += 1
    return count

def f_len_w_punc(input_arr):
    '''finding average length of sentence'''
    return len(input_arr)

def f_len_wo_punc(input_arr):
    '''finding average length of sentences without punctuation'''
    regex_str = " ".join(input_arr)
    tokens = re.findall(r"[\w]+/", regex_str)
    length = len(tokens)
    return length

def f_filler(input_arr):
    '''filler words'''
    keywords = open(wordlist_url + 'filler').read().splitlines()
    keywords = [z.lower() for z in keywords]
    count = 0
    for token in input_arr:
        token_arr = token.split("/")
        test_token = token_arr[0].lower()
        if test_token in keywords:
            count += 1
    return count

def build_basic_feat(input_arr):
    vecs = []
    for sent in input_arr:
        pre_arr = sent
        feat_arr = []
        feat_arr.append(f_first_person(pre_arr))
        feat_arr.append(f_second_person(pre_arr))
        feat_arr.append(f_second_person(pre_arr))
        feat_arr.append(f_conj(pre_arr))
        feat_arr.append(f_past_tense_verb(pre_arr))
        feat_arr.append(f_len_w_punc(pre_arr))
        feat_arr.append(f_len_wo_punc(pre_arr))
        feat_arr.append(f_filler(pre_arr))
        feat_arr.append(f_WH_count(pre_arr))
        feat_arr.append(f_ing(pre_arr))
        feat_arr.append(f_exclaim(pre_arr))
        vecs.append(feat_arr)
    return vecs