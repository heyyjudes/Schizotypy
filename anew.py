import imp
import csv
import nltk
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
            condensed_dict[keyword] = (int(row["AoA (100-700)"]), int(row["IMG"]), int(row["FAM"]))
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



if __name__ == '__main__':
    anew_dict = build_dic_anew()
    bristol_dic = build_dic_bristol()
    anewPrep = prep.PreProcessorLDA('data\\IPII_patient', dir=True)
    anewPrep.tokenize_LDA()
    anewPrep.remove_stop_LDA()
    anewPrep.stem_LDA()
    for i in range(0, len(anewPrep.stemmed_tokens)):
         print i
         val, aff, dom = calculate_score_anew(anewPrep.stemmed_tokens[i], anew_dict)
         print "Valance", val, "Affect", aff, "Dominance", dom
         aoa, img, fam = calculate_score_bristol(anewPrep.no_stop_tokens[i], bristol_dic)
         print "Age of Aquisition", aoa, "Imagability", img, "Familiarity", fam

