
import collections
import csv
import glob
import math
import nltk.probability
import nltk.tree
import numpy as np
import os
import re
import scipy.spatial.distance
import subprocess

import lib.functions
import lib.transcript
import lib.yngve


class Feature(object):
    
    def __init__(self, feature_type, name, value):
        '''Parameters:
        feature_type : string. Used to group features. E.g., "lexical", "syntactic", "semantic"
        name : string, the name of the feature. E.g., "avg_cosine_distance"
        value : float, the value of the feature. E.g., "1.0"
        '''
        self.feature_type = feature_type
        self.name = name
        self.value = value

class FeatureSet(object):
    
    def __init__(self, features=[]):
        '''Parameters:
        features : optional, list. A list of Feature objects.
        '''
        self.features = features
    
    def add(self, new_feature):
        if type(new_feature) == list:
            self.features += new_feature
        else:
            self.features += [new_feature]
    
    def get_length(self):
        '''Return the number of features in the set.'''
        return len(self.features)
    
    def __getitem__(self, index):
        '''Overload the [] operator to enable subscription.'''
        return self.features[index]
    
    def __str__(self):
        return "FeatureSet(%d features)" % (self.get_length())
    
    def __repr__(self):
        return self.__str__()    
    
    
class FeatureExtractor(object):

    # Remove stopwords (don't use the NLTK stopword list because it is too extensive and contains some verbs)
    stopwords = ['the', 'and', 'is', 'a', 'to', 'i', 'on', 'in', 'of', 'it'] # top 10, function words only
    transcript_extension = 'txt'
    
    # Initialize lexical lemmatizer
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    
    # Define variables needed for feature extraction
    inflected_verb_tags = ['VBD', 'VBG', 'VBN', 'VBZ']
    light_verbs = ["be","have","come","go","give","take","make","do","get","move","put"]
    demonstratives = ["this","that","these","those"]
    function_tags = ["DT","PRP","PRP$","WDT","WP","WP$","CC","RP","MD","IN"]
    # From Szmrecsanyi 2004
    subordinate = ["because","since","as","when","that","although","though","while","before","after","even","now","once","than","unless","until","when","whenever","where","while","who","whoever","why"] 
    
    # When checking for English words, the following tokens should be ignored:
    word_exceptions = ["n't","'m","'s","'ll","'re","'d","'ve"]
    
    # Value to use for infinity
    inf_value = 10^10
    nan_value = -1
    
    def __init__(self, source_transcript, source_transcript_fillers, utterance_sep, path_output_lu_parses, path_output_parses, 
                 parser_path, cfg_rules_path, pos_tagger_path=None, path_to_freq_norms=None, path_to_image_norms=None, 
                 path_to_dictionary=None, lu_analyzer_path=None):
        '''Parameters:
        source_transcript : list of strings. Full paths to directories containing transcripts (with no filler annotations)
        source_transcript_fillers : list of string. Full paths to a directories containing transcripts with filler annotations
        utterance_sep : string. The string that delimits utterance boundaries in the transcript
        path_lu_output_parses : string. The absolute path to a directory that will store the Lu features and parses.
        path_output_parses : string. The absolute path to a directory that will store the parse trees produced for the data.
        parser_path : string. The absolute path to a directory containing a Stanford lexparser
        cfg_rules_path : string. The absolute path to a file containing cfg productions to be extracted (one per line)
        pos_tagger_path : optional, string. Full path to a directory containing a Stanford POS tagger 
        path_to_freq_norms : optional, string. Full path to a file containing frequency norms
        path_to_image_norms : optional, string. Full path to a file containing imageability norms
        path_to_dictionary : optional, string. Full path to a file containing valid words for the language
        lu_analyzer_path : optional
        '''
        self.source_transcript_dir = source_transcript
        self.source_transcript_fillers_dir = source_transcript_fillers
        self.utterance_sep = utterance_sep
        self.output_parse_dir = path_output_parses
        self.output_lu_parse_dir = path_output_lu_parses
        self.transcript_set = lib.transcript.TranscriptSet(dataset=[])
        self.transcript_set_fillers = lib.transcript.TranscriptSet(dataset=[])
        self.pos_tagger_path = pos_tagger_path
        self.parser_path = parser_path
        self.cfg_rules_path = cfg_rules_path
        
        # Get the list of files to be processed
        for next_dir in self.source_transcript_dir:
            for filepath in glob.glob(os.path.join(next_dir, '*.' + self.transcript_extension)):
                self.transcript_set.append(lib.transcript.PlaintextTranscript(filepath=filepath, label=None, pos_tagger_path=pos_tagger_path))
        
        # Get the transcripts with fillers, if they are available
        if self.source_transcript_fillers_dir is not None:
            for next_dir in self.source_transcript_fillers_dir:
                for filepath in glob.glob(os.path.join(next_dir, '*.' + self.transcript_extension)):
                    self.transcript_set_fillers.append(lib.transcript.PlaintextTranscript(filepath=filepath, label=None, pos_tagger_path=pos_tagger_path))
        
        # Get lexical norms
        if path_to_freq_norms is not None:
            self.norms_freq = lib.functions.get_frequency_norms(path_to_freq_norms)
        else: # default
            self.norms_freq = lib.functions.get_frequency_norms()
        
        if path_to_image_norms is not None:
            self.norms_image = lib.functions.get_imageability_norms(path_to_image_norms)
        else: # default
            self.norms_image = lib.functions.get_imageability_norms()
        
        # Set up the dictionary of valid words for the language
        if path_to_dictionary is not None:
            source_dict = path_to_dictionary
        else:
            source_dict = os.path.abspath("../feature_extraction/text/american-english") # default
        with open(source_dict, 'r') as fin_dict:
            words = fin_dict.readlines()
            self.dictionary_words = set(word.strip().lower() for word in words)
        
        if lu_analyzer_path is not None:
            self.lu_analyzer_path = lu_analyzer_path
        else:
            self.lu_analyzer_path = os.path.abspath('../L2SCA-2011-10-10/')
            
    def extract(self, debug_output=None, do_lexical=True, do_syntactic=True, lexical_list=None, syntactic_list=None):
        '''Parameters:
        debug_output : optional, string. Full path to a directory to store partial results. If None,
                       no partial results are written out.
        do_lexical : optional (default=True), Boolean. If True, extract lexical features.
        do_syntactic : optional (default=True), Boolean. If True, extract syntactic features.
        lexical_list : optional (default=None), list or None. If not None, limit extracted lexical features to given list.
        syntactic_list : optional (default=None), list or None. If not None, limit extracted syntactic features to given list.
        
        Return: nothing.'''
        # Extract all features for all the transcripts
        for t in self.transcript_set:
            
            # For debug purposes, output the partial results to csv
            debug_csv = ""
            if debug_output is not None:
                debug_csv = os.path.join(debug_output, '%s.csv' % t.filename)
                
            # Only perform if we haven't computed the features yet
            if debug_output is None or not os.path.exists(debug_csv):
                
                # Get a list of lists of tokens (each row is an utterance)
                transcript_utterances = t.tokens
                pos_utterances = t.get_pos_tagged()
                
                # Get corresponding transcript with fillers, if it's available 
                # (assume filenames should match)
                transcript_utterances_fillers = None
                filter_set = [x for x in self.transcript_set_fillers if x.filename == t.filename]
                if len(filter_set) == 1:
                    transcript_utterances_fillers = filter_set[0].tokens
                
                total_words = len([token_tuple for utt in pos_utterances for token_tuple in utt])
                
                # Create a list of Feature objects, and add to transcript. Always sort in ascending order of 
                # feature name, so we get the same order for all transcripts.
                features = []
                
                # LEXICAL FEATURES (get a dict of key=feat_name, value=feat_value).
                # Then, normalize by overall transcript word counts and obtain ratios.
                if do_lexical:
                    features_lexical, sorted_lexical_names = self.extract_lexical(transcript_utterances, transcript_utterances_fillers, pos_utterances, total_words, list_features=lexical_list)
                    features_lexical = self.normalize_lexical_features(features_lexical, total_words)
                    for feat_name in sorted_lexical_names:
                        features += [Feature(feature_type="lexical", name=feat_name, value=features_lexical[feat_name])]
                
                # SYNTACTIC FEATURES (get a dict of key=feat_name, value=feat_value)
                if do_syntactic:
                    features_syntactic, sorted_syntactic_names = self.extract_syntactic(t.filepath, t.filename, transcript_utterances, syntactic_list)
                    features_syntactic = self.normalize_syntactic_features(features_syntactic, total_words)
                    for feat_name in sorted_syntactic_names:
                        features += [Feature(feature_type="syntactic", name=feat_name, value=features_syntactic[feat_name])]
                
                # Add all extracted features to transcript
                t.add_feature(features)
                
                if debug_output is not None:
                    # Assume all transcripts in the set have the same features in the same order
                    headers = [feat.name for feat in t.feature_set]
                    with open(debug_csv, 'wb') as csvfout:
                        csvwriter = csv.writer(csvfout, delimiter=',', quoting=csv.QUOTE_ALL)
                        csvwriter.writerow(headers)
                        csvwriter.writerow([feat.value for feat in t.feature_set])
            
    
    def extract_lexical(self, transcript_utterances, transcript_utterances_fillers, pos_utterances, total_words, list_features=None):
        '''Parameters:
        transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
        transcript_utterances_fillers : list of lists of strings (words); each row is a plaintext utterance containing filled pauses
        pos_utterances : list of lists of tuples of (token, POStag); each row is an utterance. No filled pauses.
        total_words : int, the total number of words in the transcript (used as normalization constant).
        list_features : optional (default=None), list or None. If not None, limit the extracted features to those in list.
        
        Return: 
        feature_dict : dict of key=feat_name, value=feat_value.
        sorted_keys : list of strings (feature names).
        '''
        feature_dict = collections.defaultdict(int)
        sorted_keys = []
        
        if list_features is None or 'cosine_distance' in list_features:
            # REPETITION
            # Build a vocab for the transcript
            fdist_vocab = nltk.probability.FreqDist([word for utt in transcript_utterances for word in utt])
            vocab_words = fdist_vocab.keys()
            for s in self.stopwords:
                if s in vocab_words:
                    vocab_words.remove(s)
            
            num_utterances = len(transcript_utterances)
            
            # Create a word vector for each utterance, N x V 
            # where N is the num of utterances and V is the vocab size
            # The vector is 1 if the vocab word is present in the utterance, 
            # 0 otherwise (i.e., one hot encoded).
            word_vectors = []
            for i, utt in enumerate(transcript_utterances):
                word_vectors.append(len(vocab_words)*[0]) # init
                for j in range(len(vocab_words)):
                    if vocab_words[j] in utt:
                        word_vectors[i][j] += 1
            
            # Calculate cosine DISTANCE between each pair of utterances in 
            # this transcript (many entries with small distances means the 
            # subject is repeating a lot of words).
            average_dist = 0.0
            min_dist = 1.0
            num_similar_00 = 0.0
            num_similar_03 = 0.0
            num_similar_05 = 0.0
            num_pairs = 0
            for i in range(num_utterances):
                for j in range(i):
                    # The norms of the vectors might be zero if the utterance contained only 
                    # stopwords which were removed above. Only compute cosine distance if the 
                    # norms are non-zero; ignore the rest.
                    norm_i, norm_j = np.linalg.norm(word_vectors[i]), np.linalg.norm(word_vectors[j])
                    if norm_i > 0 and norm_j > 0:
                        cosine_dist = scipy.spatial.distance.cosine(word_vectors[i], word_vectors[j])
                        if math.isnan(cosine_dist):
                            continue
                        average_dist += cosine_dist
                        num_pairs += 1
                        if cosine_dist < min_dist:
                            min_dist = cosine_dist
                        
                        # Try different cutoffs for similarity
                        if cosine_dist < 0.001: #similarity threshold
                            num_similar_00 +=1
                        if cosine_dist <= 0.3: #similarity threshold
                            num_similar_03 +=1
                        if cosine_dist <= 0.5: #similarity threshold
                            num_similar_05 +=1
            
            # The total number of unique utterance pairwise comparisons is <= N*(N-1)/2
            # (could be less if some utterances contain only stopwords and end up empty after
            # stopword removal).
            denom = num_pairs
            
            if denom >= 1:
                cosine_features = [average_dist * 1.0 / denom,
                                   min_dist,
                                   num_similar_00 * 1.0 / denom,
                                   num_similar_03 * 1.0 / denom,
                                   num_similar_05 * 1.0 / denom]
            else:
                # There are either no utterances or a single utterance -- no repetition occurs
                cosine_features = [self.inf_value, self.inf_value, 0, 0, 0]
            
            cosine_feature_names = ["ave_cos_dist", "min_cos_dist", "cos_cutoff_00", "cos_cutoff_03", "cos_cutoff_05"]
            for ind_feat, feat_name in enumerate(cosine_feature_names):
                feature_dict[feat_name] = cosine_features[ind_feat]
                sorted_keys += [feat_name]
        
        if list_features is None or 'fillers' in list_features:
            # FILLERS
            if transcript_utterances_fillers is not None:
                
                regex_fillers = {'fillers': re.compile(r'^(?:(?:ah)|(?:eh)|(?:er)|(?:ew)|(?:hm)|(?:mm)|(?:uh)|(?:uhm)|(?:um))$'),
                                 'um': re.compile(r'^(?:(?:uhm)|(?:um))$'),
                                 'uh': re.compile(r'^(?:(?:ah)|(?:uh))$')}
                count_fillers = collections.defaultdict(int)
                
                for utt in transcript_utterances_fillers:
                    for word in utt:
                        for filler_type in regex_fillers.keys():
                            if regex_fillers[filler_type].findall(word):
                                count_fillers[filler_type] += 1
                
                for filler_type in sorted(regex_fillers.keys()):
                    feat_val = 0
                    if filler_type in count_fillers:
                        feat_val = count_fillers[filler_type]
                    feature_dict[filler_type] = feat_val
                    sorted_keys += [filler_type]
        
        if list_features is None or 'lexical' in list_features:
            # LEXICAL
            # Filter out any punctuation tags
            regex_pos_content = re.compile(r'^[a-zA-Z$]+$')
            pos_tokens = [] # list of tuples of (token, POStag) of non-punctuation tokens
            lemmatized_tokens = [] # list of lemmatized non-punctuation tokens
            sorted_keys += ['word_length', 'NID', 
                            'nouns', 'noun_frequency', 'noun_aoa', 'noun_imageability', 'noun_familiarity',
                            'verbs', 'inflected_verbs', 'light', 'verb_frequency', 'verb_aoa', 'verb_imageability', 'verb_familiarity',
                            'function', 'pronouns', 'determiners', 'adverbs', 'adjectives', 'prepositions',
                            'coordinate', 'subordinate', 'demonstratives',
                            'frequency', 'aoa', 'imageability', 'familiarity']
            for utt in pos_utterances:
                for token_tuple in utt:
                    # If the POS tag is not that of punctuation, add to tokens
                    if regex_pos_content.findall(token_tuple[1]):
                        pos_tokens += [token_tuple]
                        
                        feature_dict['word_length'] += len(token_tuple[0])
                        
                        if token_tuple[0] not in self.dictionary_words and token_tuple[0] not in self.word_exceptions:
                            feature_dict['NID'] += 1
                        
                        # Lemmatize according to the type of the word
                        lemmatized_token = self.lemmatizer.lemmatize(token_tuple[0], lib.functions.pos_treebank2wordnet(token_tuple[1]))
                        lemmatized_tokens += [lemmatized_token]
                        
                        # Count POS tags
                        if re.match(r"^NN.*$", token_tuple[1]):
                            feature_dict['nouns'] += 1
                            
                            if token_tuple[0] in self.norms_freq:
                                feature_dict["noun_frequency"] += float(self.norms_freq[token_tuple[0]][5]) # use log10WF
                                feature_dict["noun_freq_num"] += 1
                            
                            if lemmatized_token in self.norms_image:
                                feature_dict["noun_aoa"] += float(self.norms_image[lemmatized_token][0])
                                feature_dict["noun_imageability"] += float(self.norms_image[lemmatized_token][1])
                                feature_dict["noun_familiarity"] += float(self.norms_image[lemmatized_token][2])
                                feature_dict["noun_img_num"] += 1
                            
                        elif re.match(r'^V.*$', token_tuple[1]):
                            feature_dict['verbs'] += 1
                            
                            if token_tuple[1] in self.inflected_verb_tags:
                                feature_dict['inflected_verbs'] += 1
                        
                            if lemmatized_token in self.light_verbs:
                                feature_dict['light'] += 1
                        
                            if token_tuple[0] in self.norms_freq:
                                feature_dict["verb_frequency"] += float(self.norms_freq[token_tuple[0]][5]) # use log10WF
                                feature_dict["verb_freq_num"] += 1
                                
                            if lemmatized_token in self.norms_image:
                                feature_dict["verb_aoa"] += float(self.norms_image[lemmatized_token][0])
                                feature_dict["verb_imageability"] += float(self.norms_image[lemmatized_token][1])
                                feature_dict["verb_familiarity"] += float(self.norms_image[lemmatized_token][2])
                                feature_dict["verb_img_num"] += 1
                                
                        else:
                            
                            if token_tuple[1] in self.function_tags:               
                                feature_dict['function'] += 1
            
                            if re.match(r'^PRP.*$', token_tuple[1]):
                                feature_dict['pronouns'] += 1
                            elif re.match(r"^DT$", token_tuple[1]):
                                feature_dict["determiners"] += 1
                            elif re.match(r"^RB.*$", token_tuple[1]): #adverb
                                feature_dict["adverbs"] += 1
                            elif re.match(r"^JJ.*$",token_tuple[1]): #adjective
                                feature_dict["adjectives"] += 1
                            elif re.match(r"^IN$",token_tuple[1]):
                                feature_dict["prepositions"] += 1
                            elif re.match(r"^CC$",token_tuple[1]):
                                feature_dict["coordinate"] += 1
                        
                        if token_tuple[0] in self.subordinate:
                            if token_tuple[1] in ["IN", "WRB", "WP"]:
                                feature_dict["subordinate"] += 1
                                
                        if token_tuple[0] in self.demonstratives:
                            feature_dict["demonstratives"] += 1
                        
                        if token_tuple[0] in self.norms_freq: #note: frequencies are not lemmatized
                            feature_dict["frequency"] += float(self.norms_freq[token_tuple[0]][5]) # use log10WF
                            feature_dict["freq_num"] += 1
    
                        if lemmatized_token in self.norms_image:
                            feature_dict["aoa"] += float(self.norms_image[lemmatized_token][0])
                            feature_dict["imageability"] += float(self.norms_image[lemmatized_token][1])
                            feature_dict["familiarity"] += float(self.norms_image[lemmatized_token][2])
                            feature_dict["img_num"] += 1
            
            # VOCAB RICHNESS measures
            # TTR, MATTR, Brunet, Honore
            
            # MATTR - shift a window over the transcript and compute 
            # moving average TTR over each window, then average over all windows
            for window_size in [10,20,30,40,50]:
                start = 0 
                end = window_size
                MATTR = 0
                
                feature_dict['MATTR_%d' % (window_size)] = 0
                sorted_keys += ['MATTR_%d' % (window_size)]
                while end < len(lemmatized_tokens):
                    lem_types = len(set(lemmatized_tokens[start:end]))
                    MATTR += 1.0 * lem_types / window_size
                    start += 1 # shift window one word at a time
                    end += 1
                if start > 0:
                    feature_dict['MATTR_%d' % (window_size)] = 1.0 * MATTR / start
            
            word_types = len(set(pos_tokens)) # same word with different POS = different tokens (confirm with Katie)
            fd_tokens = nltk.probability.FreqDist(pos_tokens)
            
            # Count number of tokens that occur only once in transcript
            once_words = 0
            for num in fd_tokens.values():
                if num == 1:
                    once_words += 1
            
            try:
                feature_dict["TTR"] = 1.0 * word_types / total_words
                feature_dict["brunet"] = 1.0 * total_words**(word_types**(-0.165)) # Brunet's index - Vlado
            except:
                feature_dict["TTR"] = 0
                feature_dict["brunet"] = 0
            try:
                feature_dict["honore"] = 100.0 * math.log(total_words)/(1.0-1.0*once_words/word_types) # Honore's statistic-Vlado
            except:
                feature_dict["honore"] = self.inf_value #or infinity ...? (If all words are used only once)
            sorted_keys += ['TTR', 'brunet', 'honore']
        
            # Additional ratios computed below
            sorted_keys += ['nvratio', 'prp_ratio', 'noun_ratio', 'sub_coord_ratio']
            
            # Compute verb ratios, and noun to verb ratio
            if feature_dict["verbs"] > 0:
                feature_dict["nvratio"] = 1.0 * feature_dict["nouns"] / feature_dict["verbs"]
                feature_dict["light"] = 1.0 * feature_dict["light"] / feature_dict["verbs"]
                feature_dict["inflected_verbs"] = 1.0 * feature_dict["inflected_verbs"] / feature_dict["verbs"] 
            else:
                if feature_dict["nouns"] > 0:
                    feature_dict["nvratio"] = self.inf_value
                else:
                    feature_dict["nvratio"] = self.nan_value
                feature_dict["light"] = 0
                feature_dict["inflected_verbs"] = 0
            
            # Compute noun ratios (pronouns to pronoun+nouns, and nouns to noun+verb)
            if feature_dict["nouns"] > 0:
                feature_dict["prp_ratio"] = 1.0 * feature_dict["pronouns"] / (feature_dict["pronouns"] + feature_dict["nouns"])
                feature_dict["noun_ratio"] = 1.0 * feature_dict["nouns"] / (feature_dict["verbs"] + feature_dict["nouns"])
            else:
                if feature_dict["pronouns"] > 0:
                    feature_dict["prp_ratio"] = 1.0 * feature_dict["pronouns"] / (feature_dict["pronouns"] + feature_dict["nouns"])
                else:
                    feature_dict["prp_ratio"] = self.nan_value # NaN? 0/0 - no nouns and no pronouns
                    
                if feature_dict["verbs"] > 0:
                    feature_dict["noun_ratio"] = 1.0 * feature_dict["nouns"]/(feature_dict["verbs"] + feature_dict["nouns"])
                else:
                    feature_dict["noun_ratio"] = self.nan_value # NaN? 0/0 - no nouns and no verbs
            
            # Compute conjunction ratios        
            if feature_dict["coordinate"] > 0:
                feature_dict["sub_coord_ratio"] = 1.0 * feature_dict["subordinate"] / feature_dict["coordinate"]
            else:
                if feature_dict['subordinate'] > 0:
                    feature_dict["sub_coord_ratio"] = self.inf_value
                else:
                    feature_dict['sub_coord_ratio'] = self.nan_value # NaN? 0/0 - no subord and no coord conjunctions
            
            # Normalize all age of acquisition, imageability, familiarity norms
            if feature_dict["img_num"] > 0:
                feature_dict["aoa"] = 1.0 * feature_dict["aoa"] / feature_dict["img_num"]    
                feature_dict["imageability"] = 1.0 * feature_dict["imageability"] / feature_dict["img_num"]    
                feature_dict["familiarity"] = 1.0 * feature_dict["familiarity"] / feature_dict["img_num"]    
            else: # no words with imageability norms
                feature_dict["aoa"] = self.nan_value
                feature_dict["imageability"] = self.nan_value  
                feature_dict["familiarity"] = self.nan_value
            
            # Normalize all age of acquisition, imageability, familiarity norms for nouns
            if feature_dict["noun_img_num"] > 0:
                feature_dict["noun_aoa"] = 1.0 * feature_dict["noun_aoa"] / feature_dict["noun_img_num"]    
                feature_dict["noun_imageability"] = 1.0 * feature_dict["noun_imageability"] / feature_dict["noun_img_num"]    
                feature_dict["noun_familiarity"] = 1.0 * feature_dict["noun_familiarity"] / feature_dict["noun_img_num"]    
            else:
                feature_dict["noun_aoa"] = self.nan_value
                feature_dict["noun_imageability"] = self.nan_value   
                feature_dict["noun_familiarity"] = self.nan_value
        
            # Normalize all age of acquisition, imageability, familiarity norms for verbs
            if feature_dict["verb_img_num"] > 0:
                feature_dict["verb_aoa"] = 1.0 * feature_dict["verb_aoa"] / feature_dict["verb_img_num"]    
                feature_dict["verb_imageability"] = 1.0 * feature_dict["verb_imageability"] / feature_dict["verb_img_num"]    
                feature_dict["verb_familiarity"] = 1.0 * feature_dict["verb_familiarity"] / feature_dict["verb_img_num"]    
            else:
                feature_dict["verb_aoa"] = self.nan_value
                feature_dict["verb_imageability"] = self.nan_value   
                feature_dict["verb_familiarity"] = self.nan_value
            
            # Normalize frequency norms
            if feature_dict["freq_num"] > 0:
                feature_dict["frequency"] = 1.0 * feature_dict["frequency"] / feature_dict["freq_num"]    
            else:
                feature_dict["frequency"]= self.nan_value
                
            # Normalize frequency norms for nouns
            if feature_dict["noun_freq_num"] > 0:
                feature_dict["noun_frequency"] = 1.0 * feature_dict["noun_frequency"] / feature_dict["noun_freq_num"]
            else:
                feature_dict["noun_frequency"] = self.nan_value
            
            # Normalize frequency norms for verbs
            if feature_dict["verb_freq_num"] > 0:    
                feature_dict["verb_frequency"] = 1.0 * feature_dict["verb_frequency"] / feature_dict["verb_freq_num"]
            else:
                feature_dict["verb_frequency"] = self.nan_value
        
        return feature_dict, sorted_keys
        
        
    def extract_syntactic(self, transcript_filepath, transcript_filename, transcript_utterances, list_features=None):
        '''Parameters:
        transcript_filepath : string. The absolute path to the file containing the transcript.
        transcript_filename : string. The filename of the file containing the transcript.
        transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
        list_features : optional (default=None), list or None. If not None, limit the extracted features to those in list.
        
        Return: 
        feature_dict : dict of key=feat_name, value=feat_value.
        sorted_keys : list of strings (feature names).
        '''
        feature_dict = {}
        sorted_keys = []
        
        if list_features is None or 'lu_complexity' in list_features:
            # LU'S COMPLEXITY FEATURES
            # Run the Lu complexity analyzer script which produces a csv file with the feature values,
            # and a CFG parse of the text. Then, read the features into the dictionary.
            subprocess.call(['python', os.path.join(self.lu_analyzer_path, 'analyzeText.py'), 
                             transcript_filepath, os.path.join(self.output_lu_parse_dir, transcript_filename)],
                            cwd=self.lu_analyzer_path)
            
            with open(os.path.join(self.output_lu_parse_dir, transcript_filename),'r') as fin_lu:
                headers = fin_lu.readline().strip().split(',')
                lu_features = fin_lu.readline().strip().split(',')
                for i in range(1,len(headers)):
                    feature_dict[headers[i]] = float(lu_features[i])
                    sorted_keys += [headers[i]]
        
        if list_features is None or 'parsetrees' in list_features:
            # TREE MEASURES
            # Build stanford CFG and dependency parses, only if they don't exist already
            target_parse = os.path.join(self.output_parse_dir, transcript_filename + '.parse')
            if not os.path.exists(target_parse):
                # "oneline" parse (parse for one utterance per line)
                with open(target_parse, 'w') as fout:
                    subprocess.call([os.path.join(self.parser_path, 'lexparser_oneline.sh'), transcript_filepath], stdout = fout)
                
            # "penn,typedDependencies" parse
            target_depparse = os.path.join(self.output_parse_dir, transcript_filename + '.depparse')
            if not os.path.exists(target_depparse):
                with open(target_depparse, 'w') as fout:
                    subprocess.call([os.path.join(self.parser_path, 'lexparser_dep.sh'), transcript_filepath], stdout = fout)
            
            with open(target_parse, 'r') as fin:
                treelist = fin.readlines()
                
                maxdepth = 0.0
                totaldepth = 0.0
                meandepth = 0.0
                treeheight = 0.0
                numtrees = 0
                
                ###from Jed
                # To read in the parse trees into nltk tree objects, expect the 'oneline' 
                # format from the stanford parser (one utterance tree per line).
                yc = lib.yngve.Yngve_calculator()
                for utterance_tree in treelist:
                    if utterance_tree:
                        thistree = utterance_tree # read parsed tree from parser-processed file
                        numtrees += 1
                        pt = nltk.tree.ParentedTree.fromstring(thistree) #convert to nltk tree format
                        st = list(pt.subtrees()) #extract list of all sub trees in tree
                        nodelist = []
                        for subt in st:
                            nodelist.append(subt.label())  # make list of all node labels for subtrees
                        
                        Snodes = nodelist.count('S') + nodelist.count('SQ') + nodelist.count('SINV')#count how many nodes are "S" (clauses)
                        
                        # A list of the Yngve depth (int) for each terminal in the tree
                        depthlist = yc.make_depth_list(pt,[])  # computes depth, need to pass it an empty list
                        depthlist = depthlist[:-1] # the last terminal is a punctuation mark, ignore it
                        if depthlist:
                            maxdepth += max(depthlist)
                            totaldepth += sum(depthlist)
                            if len(depthlist) > 0:
                                meandepth += 1.0*sum(depthlist)/len(depthlist)
                        treeheight += pt.height()
                
                if numtrees > 0:
                    feature_dict['maxdepth'] = maxdepth / numtrees # or should it be max overall?
                    feature_dict['totaldepth'] = totaldepth / numtrees
                    feature_dict['meandepth'] = meandepth / numtrees
                    feature_dict['treeheight'] = treeheight / numtrees
                else:
                    feature_dict['maxdepth'] = 0
                    feature_dict['totaldepth'] = 0
                    feature_dict['meandepth'] = 0
                    feature_dict['treeheight'] = 0
                sorted_keys += ['maxdepth', 'totaldepth', 'meandepth', 'treeheight']
                
                
                # CFG MEASURES
                # Count frequency of different CFG constituents, using the 
                # constructed parse tree
                        
                totNP = 0
                totVP = 0
                totPP = 0
                lenNP = 0
                lenVP = 0
                lenPP = 0
                total_length = 0
                prod_nonlexical = []
                
                # List of rules to look for
                with open(self.cfg_rules_path, 'r') as fin:
                    rules = fin.read()
                    top_rules = rules.strip().split('\n')
                
                    for utterance_tree in treelist:
                        if utterance_tree:
                            # Convert to unicode to prevent errors when there 
                            # are non-ascii characters
                            utterance_tree = unicode(utterance_tree, 'utf-8')
                            t = nltk.tree.Tree.fromstring(utterance_tree)
                            prods = t.productions()
                            for p in prods:
                                if p.is_nonlexical():
                                    prod_nonlexical.append(re.sub(" ","_",unicode(p)))
                            
                            # Counting phrase types
                            for st in t.subtrees():
                                if str(st).startswith("(NP"):
                                    lenNP += len(st.leaves())
                                    totNP += 1
                                elif str(st).startswith("(VP"):
                                    lenVP += len(st.leaves())
                                    totVP += 1
                                elif str(st).startswith("(PP"):
                                    lenPP += len(st.leaves())
                                    totPP += 1
                                
                            sent_length = len(t.leaves())
                            total_length += sent_length
                
                    if total_length > 0:
                        feature_dict["PP_type_prop"] = 1.0*lenPP/total_length
                        feature_dict["VP_type_prop"] = 1.0*lenVP/total_length
                        feature_dict["NP_type_prop"] = 1.0*lenNP/total_length
                    
                        feature_dict["PP_type_rate"] = 1.0*totPP/total_length
                        feature_dict["VP_type_rate"] = 1.0*totVP/total_length
                        feature_dict["NP_type_rate"] = 1.0*totNP/total_length
                    else:
                        feature_dict["PP_type_prop"] = 0
                        feature_dict["VP_type_prop"] = 0
                        feature_dict["NP_type_prop"] = 0
                    
                        feature_dict["PP_type_rate"] = 0
                        feature_dict["VP_type_rate"] = 0
                        feature_dict["NP_type_rate"] = 0
     
                    try:
                        feature_dict["average_PP_length"] = 1.0*lenPP/totPP
                    except:
                        feature_dict["average_PP_length"] = 0
                    try:
                        feature_dict["average_VP_length"] = 1.0*lenVP/totVP
                    except:
                        feature_dict["average_VP_length"] = 0
                    try:
                        feature_dict["average_NP_length"] = 1.0*lenNP/totNP
                    except:
                        feature_dict["average_NP_length"] = 0
                    
                    sorted_keys += ['PP_type_prop', 'VP_type_prop', 'NP_type_prop',
                                    'PP_type_rate', 'VP_type_rate', 'NP_type_rate', 
                                    'average_PP_length', 'average_VP_length', 'average_NP_length']
                    
                    # Normalize by number of productions
                    num_productions = len(prod_nonlexical) 
                    fdist = nltk.probability.FreqDist(prod_nonlexical)
                    
                    for prod_rule in top_rules: # need this to ensure we always get same number of CFG features
                        if prod_rule in fdist: 
                            feature_dict[prod_rule] = 1.0 * fdist[prod_rule] / num_productions
                        else:
                            feature_dict[prod_rule] = 0.0
                        sorted_keys += [prod_rule]
                    
                
        return feature_dict, sorted_keys
    
    def normalize_syntactic_features(self, feature_dict, normalization_factor):
        
        # NORMALIZATION by total number of words
        norm_by_total_words = ["T","C","CN","CP","CT","VP","DC"]
        
        if normalization_factor > 0:
            for feat_name in norm_by_total_words:
                if feat_name in feature_dict:
                    feature_dict[feat_name] = feature_dict[feat_name] * 1.0 / normalization_factor
        return feature_dict
       
    def normalize_lexical_features(self, feature_dict, normalization_factor):
        
        # NORMALIZATION of pos counts
        # feature IDs that should be normalized by the total number of words in the transcript
        norm_by_total_words = ["nouns","verbs","pronouns","word_length","function","demonstratives",
                               "prepositions","adverbs","adjectives",
                               "determiners","coordinate","subordinate","NID","um","uh","fillers",
                               "T","C","CN","CP","CT","VP","DC"]
        
        if normalization_factor > 0:
            
            # Normalize all features by the total number of words in transcript
            for feat_name in norm_by_total_words:
                if feat_name in feature_dict:
                    feature_dict[feat_name] = 1.0 * feature_dict[feat_name] / normalization_factor
            
        return feature_dict
    
    def write_to_csv(self, output_csv):
        '''Parameters:
        output_csv : string. The full path and filename of the output CSV where we store the features.
                     The first line contains the feature names, and every line thereafter corresponds to 
                     the features for one transcript. Num rows = num transcripts, num cols = num features.
        Return: nothing.
        '''
        # Assume all transcripts in the set have the same features in the same order
        if self.transcript_set.get_length() > 0 and self.transcript_set[0].feature_set:
            headers = ['FileID'] + [feat.name for feat in self.transcript_set[0].feature_set]
            with open(output_csv, 'wb') as csvfout:
                csvwriter = csv.writer(csvfout, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(headers)
                for t in self.transcript_set:
                    csvwriter.writerow([lib.functions.get_fileid(t.filename)] + [feat.value for feat in t.feature_set])
                