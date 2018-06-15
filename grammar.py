import os
# os.environ['CLASSPATH'] = "C:\\Users\\heyyj\\PycharmProjects\\stanford-parser-full-2016-10-31\\stanford-parser.jar"
# os.environ['JAVAHOME'] = "C:\\Program Files\\Java\\jre1.8.0_112\\bin"
from nltk.parse.stanford import StanfordParser
import nltk
import NLP_lib
import re
import numpy as np
# stanford_parser_dir = "C:\\Users\\heyyj\\PycharmProjects\\stanford-parser-full-2016-10-31\\"
# eng_model_path = stanford_parser_dir + "edu\\stanford\\nlp\\models\\lexparser\\englishRNN.ser.gz"
# my_path_to_models_jar = stanford_parser_dir + "stanford-parser-3.7.0-models.jar"
# my_path_to_jar = stanford_parser_dir + "stanford-parser.jar"
#
# parser = StanfordParser(path_to_models_jar=my_path_to_models_jar, path_to_jar=my_path_to_jar)

yc = NLP_lib.yngve.Yngve_calculator()

def g_err_count(input_arr):
    pass

def g_emb_depth(sent):
    '''
    :param input_arr: list of tree
    :return: height of tree
    '''
    return sent.height()

def g_NP_complex(sent):
    '''
    counting average number of leaves in NP clause
    :param input_arr: array of trees
    :return: NP rate, avg NP len,
    '''
    length = 0.0
    avg_length = 0.0
    num_NP = 0
    sent_len = len(sent.leaves())
    #for sent in input_arr:
    for s in sent.subtrees():
        clause_arr = re.findall(r"\w+", str(s))
        if len(clause_arr) > 0:
            if clause_arr[0] == "NP":
                leaves = s.leaves()
                num_NP += 1
                length += float(len(leaves))

    if num_NP > 0:
        avg_length = length / num_NP

    return [avg_length, num_NP/sent_len, length/sent_len]

def g_VP_complex(sent):
    '''
    counting average number of leaves in VP clause
    :param input_arr: array of trees
    :return: original tree and average length of VP clauses
    '''
    length = 0.0
    avg_length = 0.0
    num_VP = 0
    sent_len = len(sent.leaves())
    for s in sent.subtrees():
        clause_arr = re.findall(r"\w+", str(s))
        if len(clause_arr) > 0:
            if clause_arr[0] == "VP":
                leaves = s.leaves()
                num_VP += 1
                length += float(len(leaves))
    if num_VP > 0:
        avg_length = length / num_VP
    return [avg_length, num_VP/sent_len, length/sent_len]

def g_PP_complex(sent):
    '''
    counting average number of leaves in PP clause
    :param input_arr: array of trees
    :return: original tree and average length of PP clauses
    '''
    length = 0.0
    avg_length = 0.0
    num_PP = 0
    sent_len = len(sent.leaves())
    for s in sent.subtrees():
        clause_arr = re.findall(r"\w+", str(s))
        if len(clause_arr) > 0:
            if clause_arr[0] == "PP":
                leaves = s.leaves()
                num_PP += 1
                length += float(len(leaves))
    if num_PP > 0:
        avg_length = length / num_PP
    return [avg_length, num_PP/sent_len, length/sent_len]

def g_right_left(sent):
    '''
    calculate unequal branching of tree
    :param sent:
    :return:
    '''
    tree_arr = []
    for st in sent.subtrees():
        tree_arr.append(st)
    right_len = len(tree_arr[1].leaves())
    left_len = len(tree_arr[2].leaves())
    total_len = len(sent.leaves())
    return 1.0*(right_len-left_len)/total_len


def yngve_depth(sent):
    pt = nltk.tree.ParentedTree.fromstring(str(sent))
    depth_list = yc.make_depth_list(pt, [] )
    max_depth = max(depth_list)
    total_depth = sum(depth_list)
    mean_depth = 1.0*sum(depth_list)/len(depth_list)

    return [max_depth, total_depth, mean_depth]

def reformat_production(input_str):
    new_str = re.sub(r"PRP\$", "Det", input_str)
    new_str = re.sub(r"CD\$", "Det", new_str)
    return new_str

def build_complexity_features(input_arr, filename):
    '''
    array of untagged tokens
    :param input_arr: sentence
    :return:
    '''

    vecs = []

    text_str = filename + 'txt'
    parse_file = open(text_str, 'w')
    for sent in input_arr:
        feat_vec = []
        if len(sent) > 0:
            input_str = " ".join(sent)
            parsed_sent = parser.raw_parse(input_str)
            for line in parsed_sent:
                parse_file.write(str(parsed_sent))
                parse_file.write('\n')
                feat_vec += g_NP_complex(line) #3 feat
                feat_vec += g_VP_complex(line) #3 feat
                feat_vec += g_PP_complex(line) #3 feat
                feat_vec += yngve_depth(line) #3 feat
                emb_len = g_emb_depth(line) #1 feat
                feat_vec.append(emb_len)
                feat_vec.append(g_right_left(line))

        else:
            feat_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print feat_vec
        vecs.append(feat_vec)
        vec_arr = np.asarray(vecs)
        np.save(filename, vec_arr)
    return vecs

if __name__ == "__main__":
    one = ("I shot and killed an elephant in my pajamas")
    sent = one.split()
    print sent
    parsed_sent = parser.raw_parse(one)
    for sent in parsed_sent:
        print g_right_left(sent)
        np_len = g_NP_complex(sent)
        print np_len
        vp_len = g_VP_complex(sent)
        print vp_len
        pp_len = g_PP_complex(sent)
        print pp_len
        print yngve_depth(sent)


    #for line in parsed_sent:
        # print line
        # print line.height()
        # line.draw()
        # for s in line.subtrees():
        #     print s
        # cfg_arr = []
        # for element in line.productions():
        #     cfg_arr.append(str(element))
        # cfg_str = "\n".join(cfg_arr)
        # cfg_str = reformat_production(cfg_str)
        # print cfg_str

        # grammar1 = nltk.CFG.fromstring(cfg_str)
        # rd_parser = nltk.RecursiveDescentParser(grammar1)
        #
        # for tree in rd_parser.parse(sent):
        #     print tree
