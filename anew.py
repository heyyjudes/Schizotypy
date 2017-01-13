import imp
import csv
import nltk
import numpy as np
from logreg import run_LR
from svm import run_SVM
from sklearn.preprocessing import scale

prep = imp.load_source('preprocess', 'C:\\Users\\heyyj\\PycharmProjects\\NLP_lib\\preprocess.py')

def build_dic_bristol():
    input_file = csv.DictReader(open('data\\BristolNorms\\BristolNorms+GilhoolyLogie.csv', 'r'))
    condensed_dict = {}
    porter = nltk.PorterStemmer()
    for row in input_file:
        keyword = row["WORD"]
        # keyword = porter.stem(row["WORD"])
        # if keyword in condensed_dict.keys():
        #     print 'WARNING Stemming Duplicate Found', keyword
        if keyword:
            condensed_dict[keyword] = (float(row["AoA (100-700)"])/100, float(row["IMG"])/100, float(row["FAM"])/100)
    return condensed_dict

def calculate_score_bristol(unstemmed_tokens, dict):
    b_count = total_count = avg_aoa = avg_img = avg_fam = 0
    for phrase in unstemmed_tokens:
        for word in phrase:
            if word in dict.keys():
                aoa, img, fam = dict[word]
                b_count += 1
                avg_aoa += aoa
                avg_img += img
                avg_fam += fam
            total_count +=1
    avg_aoa = avg_aoa/b_count
    avg_img = avg_img/b_count
    avg_fam = avg_fam/b_count
    print "total bristol words", b_count
    print "total words", total_count
    print "bristol percentage", b_count*1.0/total_count
    return avg_aoa, avg_img, avg_fam

def build_dic_anew():
    input_file = csv.DictReader(open('data\\ANEW\\Ratings_Warriner_et_al.csv', 'r'))
    condensed_dict = {}
    porter = nltk.PorterStemmer()
    for row in input_file:
        keyword = porter.stem(row["Word"])
        condensed_dict[keyword] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]), float(row["D.Mean.Sum"]))
    return condensed_dict

def calculate_score_anew(tokens, dict):
    anew_count = total_count = avg_val = avg_aff = avg_dom = 0
    for phrase in tokens:
        for word in phrase:
            if word in dict:
                val, aff, dom = dict[word]
                anew_count+=1
                avg_val += val
                avg_aff += aff
                avg_dom += dom
            total_count += 1

    avg_val = avg_val/anew_count
    avg_aff = avg_aff/anew_count
    avg_dom = avg_dom/anew_count
    print "total anew", anew_count
    print "total words", total_count
    print "anew percentage", anew_count*1.0/total_count
    return avg_val, avg_aff, avg_dom

def build_model_vecs_sent(process, dict, type):
    y_set = None
    vecs_set = None
    if type == "ANEW":
        patient_arr = process.stemmed_tokens
    else:
        patient_arr = process.no_stop_tokens
    assert(len(process.stemmed_tokens) == len(process.target_arr))
    for i in range(0, len(patient_arr)):
        target = np.zeros((1,1))
        target[0][0] = process.target_arr[i]
        doc = patient_arr[i]
        for sentence in doc:
            result = np.zeros((1, 3))
            anew_words = 0
            for word in sentence:
                if word in dict:
                    val, aff, dom = dict[word]
                    result[0][0] += val
                    result[0][1] += aff
                    result[0][2] += dom
                    anew_words += 1
            if anew_words != 0:
                result /= anew_words
                if vecs_set == None:
                    vecs_set = result
                    y_set = target
                else:
                    vecs_set = np.concatenate((vecs_set, result))
                    y_set = np.concatenate((y_set, target))
    vecs_set = scale(vecs_set)
    return vecs_set, y_set

def build_model_vecs_patient(process, dict, type):
    y_set = None
    vecs_set = None
    if type == "ANEW":
        patient_arr = process.stemmed_tokens
    else:
        patient_arr = process.no_stop_tokens
    assert(len(process.stemmed_tokens) == len(process.target_arr))
    for i in range(0, len(patient_arr)):
        target = np.zeros((1,1))
        target[0][0] = process.target_arr[i]
        doc = patient_arr[i]
        result = np.zeros((1, 3))
        anew_words = 0
        for sentence in doc:
            for word in sentence:
                if word in dict:
                    val, aff, dom = dict[word]
                    result[0][0] += val
                    result[0][1] += aff
                    result[0][2] += dom
                    anew_words += 1
        if anew_words != 0:
            result /= anew_words
            if vecs_set == None:
                vecs_set = result
                y_set = target
            else:
                vecs_set = np.concatenate((vecs_set, result))
                y_set = np.concatenate((y_set, target))
    vecs_set = scale(vecs_set)
    return vecs_set, y_set

def build_model_vecs_comb_sent(process, anew_dict, bristol_dict):
    y_set = None
    vecs_set = None
    assert (len(process.stemmed_tokens) == len(process.target_arr))
    for i in range(0, len(process.stemmed_tokens)):
        target = np.zeros((1, 1))
        target[0][0] = process.target_arr[i]
        doc = process.stemmed_tokens[i]
        for sentence in doc:
            result = np.zeros((1, 6))
            anew_words = 0
            b_count = 0
            for word in sentence:
                if word in anew_dict.keys():
                    val, aff, dom = anew_dict[word]
                    result[0][0] += val
                    result[0][1] += aff
                    result[0][2] += dom
                    anew_words += 1
                if word in bristol_dict.keys():
                    aoa, img, fam = bristol_dict[word]
                    b_count += 1
                    result[0][3] += aoa
                    result[0][4] += img
                    result[0][5] += fam
            if anew_words > 5:
                result[0][0] /= anew_words
                result[0][1] /= anew_words
                result[0][2] /= anew_words
            if b_count > 5:
                result[0][3] /= b_count
                result[0][4] /= b_count
                result[0][5] /= b_count
            if (anew_words > 5) or (b_count > 5):
                if vecs_set == None:
                    vecs_set = result
                    y_set = target
                else:
                    vecs_set = np.concatenate((vecs_set, result))
                    y_set = np.concatenate((y_set, target))
    vecs_set = scale(vecs_set)
    return vecs_set, y_set

if __name__ == '__main__':
    anew_dict = build_dic_anew()
    bristol_dic = build_dic_bristol()
    anewPrep = prep.PreProcessorANEW('data\\IPII_patient', dir=True)
    anewPrep.tokenize_LDA()
    anewPrep.remove_stop_LDA()
    anewPrep.stem_LDA()
    anewPrep.doc_term_matrix()

    anewPrep.build_labels()
    # for i in range(0, len(anewPrep.stemmed_tokens)):
    #      print i
    #      val, aff, dom = calculate_score_anew(anewPrep.stemmed_tokens[i], anew_dict)
    #      print "Valance", val, "Affect", aff, "Dominance", dom
    #      aoa, img, fam = calculate_score_bristol(anewPrep.no_stop_tokens[i], bristol_dic)
    #      print "Age of Aquisition", aoa, "Imagability", img, "Familiarity", fam

    x, y =  build_model_vecs_comb_sent(anewPrep, anew_dict, bristol_dic)
    run_LR(x,y)
    run_SVM(x, y)


