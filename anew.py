import csv
import nltk
def build_dic():
    input_file = csv.DictReader(open('data\\ANEW\\Ratings_Warriner_et_al.csv', 'r'))
    condensed_dict = {}
    porter = nltk.PorterStemmer()
    for row in input_file:
        keyword = porter.stem(row["Word"])
        condensed_dict[keyword] = (row["V.Mean.Sum"], row["A.Mean.Sum"], row["D.Mean.Sum"])

    return condensed_dict
if __name__ == '__main__':
    anew_dict = build_dic()