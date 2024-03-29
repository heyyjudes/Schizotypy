from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit
import numpy as np
from logreg import run_logreg
from svm import train_svm


def cleanText(corpus):
    corpus = [z.lower().replace('\n', '').split() for z in corpus]
    return corpus


#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def w2v_train_scale(n_dim, reddit_w2v, x_train, x_test):
    train_vecs = np.concatenate([buildWordVector(z, n_dim, reddit_w2v) for z in x_train])
    train_vecs = scale(train_vecs)

    # Train word2vec on test tweets
    reddit_w2v.train(x_test)
    return train_vecs

if __name__ == "__main__":

    print('a. fetching data')
    with open('data//f_risk_ear.txt', 'r') as infile:
        dep_posts = infile.readlines()

    with open('data//f_control_ear.txt', 'r') as infile:
        reg_posts = infile.readlines()

    n_dim = 300

    y = np.concatenate((np.ones(len(reg_posts)), np.zeros(len(dep_posts))))
    x = np.concatenate((reg_posts, dep_posts))


    print('b. initializing')
    rs = ShuffleSplit(n_splits=5, test_size=.20)
    rs.get_n_splits(x)
    split = 0
    for train_index, test_index in rs.split(x):
        print "split", split
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print y_test.shape

        x_train = cleanText(x_train)
        x_test = cleanText(x_test)

        # Initialize model and build vocab
        reddit_w2v = Word2Vec(size=n_dim, min_count=10)
        reddit_w2v.build_vocab(x_train)

        print('c. training model')
        #Train the model over train_reviews (this may take several minutes)
        reddit_w2v.train(x_train)

        print('d. scaling')
        train_vecs = w2v_train_scale(n_dim, reddit_w2v, x_train, x_test)

        #Build test tweet vectors then scale
        test_vecs = np.concatenate([buildWordVector(z, n_dim, reddit_w2v) for z in x_test])
        test_vecs = scale(test_vecs)

        print('e. logistical regression')
        #Use classification algorithm (i.e. Stochastic Logistic Regression) on training set, then assess model performance on test set
        run_logreg(train_vecs, test_vecs, y_train, y_test)

        print('f. svm')
        train_svm(train_vecs, test_vecs, y_train, y_test)