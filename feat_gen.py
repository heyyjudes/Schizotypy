from anew import *
import numpy as np
from logreg import run_LR
from svm import run_SVM
from sklearn.preprocessing import scale
import NLP_lib.preprocess
import basic
import grammar

class FeatureGenerator:
    pos_text = []
    neg_text = []
    feat_vec = []
    target_vec = []
    pre_process = None

    def set_texts(self, input_pos, input_neg):
        with open(input_pos, 'r') as infile:
            self.pos_text = infile.readlines()
        with open(input_pos, 'r') as infile:
            self.neg_text = infile.readlines()

    def set_preprocess(self, process):
        self.pre_process = process

    def build_ANEW(self):
        a_dict = build_dic_anew()
        cltr_vecs = build_anew_feat(self.pre_process, a_dict, 'ANEW', self.pre_process.cltr_token)
        risk_vecs = build_anew_feat(self.pre_process, a_dict, 'ANEW', self.pre_process.risk_token)
        feat_vec = np.concatenate((cltr_vecs, risk_vecs))
        target_vec = np.concatenate((np.ones(len(cltr_vecs)), np.zeros(len(risk_vecs))))
        if len(self.target_vec) == 0:
            self.target_vec = target_vec
        if len(self.feat_vec) == 0:
            self.feat_vec = feat_vec
        else:
            assert(len(self.feat_vec) == len(feat_vec))
            comb_list = np.concatenate((self.feat_vec, feat_vec), axis=1)
            #updating feature vec with updated leave as np array for now
            self.feat_vec = comb_list

    def build_Bristol(self):
        b_dict = build_dic_bristol()
        cltr_vecs = build_anew_feat(self.pre_process, b_dict, 'Bristol', self.pre_process.cltr_token)
        risk_vecs = build_anew_feat(self.pre_process, b_dict, 'Bristol', self.pre_process.risk_token)
        feat_vec = np.concatenate((cltr_vecs, risk_vecs))
        target_vec = np.concatenate((np.ones(len(cltr_vecs)), np.zeros(len(risk_vecs))))
        if len(self.target_vec) == 0:
            self.target_vec = target_vec
        if len(self.feat_vec) == 0:
            self.feat_vec = feat_vec
        else:
            assert(len(self.feat_vec) == len(feat_vec))
            comb_list = np.concatenate((self.feat_vec, feat_vec), axis=1)
            #updating feature vec with updated leave as np array for now
            self.feat_vec = comb_list

    def build_basic(self):

        cltr_vecs = basic.build_basic_feat(self.pre_process.cltr_tagged_token)
        risk_vecs = basic.build_basic_feat(self.pre_process.risk_tagged_token)

        print "basic control", len(cltr_vecs)
        print "basic risk", len(risk_vecs)
        feat_vec = np.concatenate((cltr_vecs, risk_vecs))
        target_vec = np.concatenate((np.ones(len(cltr_vecs)), np.zeros(len(risk_vecs))))
        if len(self.target_vec) == 0:
            self.target_vec = target_vec
        if len(self.feat_vec) == 0:
            self.feat_vec = feat_vec
        else:
            assert (len(self.feat_vec) == len(feat_vec))
            comb_list = np.concatenate((self.feat_vec, feat_vec), axis=1)
            # updating feature vec with updated leave as np array for now
            self.feat_vec = comb_list

    def build_complexity(self):
        assert(self.pre_process != None)
        #cltr_vecs = grammar.build_complexity_features(self.pre_process.cltr_token, 'cltr')
        #risk_vecs = grammar.build_complexity_features(self.pre_process.risk_token, 'risk')
        cltr_vecs = np.load('cltr.npy')
        risk_vecs = np.load('risk.npy')
        print "grammar control", len(cltr_vecs)
        print "grammar risk", len(risk_vecs)
        feat_vec = np.concatenate((cltr_vecs, risk_vecs))
        target_vec = np.concatenate((np.ones(len(cltr_vecs)), np.zeros(len(risk_vecs))))
        if len(self.target_vec) == 0:
            self.target_vec = target_vec
        if len(self.feat_vec) == 0:
            self.feat_vec = feat_vec
        else:
            print len(self.feat_vec)
            print len(feat_vec)
            assert (len(self.feat_vec) == len(feat_vec))
            comb_list = np.concatenate((self.feat_vec, feat_vec), axis=1)
            # updating feature vec with updated leave as np array for now
            self.feat_vec = comb_list

    def remove_null_and_scale(self):
        new_arr = []
        new_tar = []
        empty = 0
        tar_list = self.target_vec.tolist()
        m, n = self.feat_vec.shape
        for i in range(0 , m):
            non_zero = np.count_nonzero(self.feat_vec[i])
            if non_zero != 0:
                new_arr.append(self.feat_vec[i])
                new_tar.append(tar_list[i])
        self.feat_vec = scale(np.array(new_arr))
        self.target_vec = np.array(new_tar)

if __name__ == "__main__":
    #preprocessing
    anewPrep = NLP_lib.preprocess.PreProcessorFG('data\\f_control.txt', 'data\\f_risk.txt')

    anewPrep.process()
    print len(anewPrep.cltr_sent)
    print len(anewPrep.cltr_token)
    print len(anewPrep.cltr_tagged_token)
    print len(anewPrep.risk_sent)
    print len(anewPrep.risk_token)
    print len(anewPrep.risk_tagged_token)


    #build features
    feat_gen = FeatureGenerator()
    feat_gen.set_preprocess(anewPrep)

    #feat_gen.build_ANEW()
    #feat_gen.build_Bristol()
    #feat_gen.build_basic()
    #feat_gen.build_complexity()

    #scaling
    #feat_gen.remove_null_and_scale()

    #run learning
    run_LR(feat_gen.feat_vec, feat_gen.target_vec)
    run_SVM(feat_gen.feat_vec, feat_gen.target_vec)


