import sklearn as sk
import sklearn.feature_extraction.text
import sklearn.svm
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.metrics
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np

def generate_features(train_X, val_X, test_X, max_ngram = 3):
    train_texts = []
    for l in train_X:
        train_texts.append(' '.join(l))
    
    val_texts = []
    for m in val_X:
        val_texts.append(' '.join(m))
        
    test_texts = []
    for n in test_X:
        test_texts.append(' '.join(n))
    
    generator = sklearn.feature_extraction.text.CountVectorizer(ngram_range = (1, max_ngram), analyzer = 'word')
    generator.fit(train_texts + val_texts + test_texts)
    train_texts_features = generator.transform(train_texts)
    val_texts_features = generator.transform(val_texts)
    test_texts_features = generator.transform(test_texts)
    return train_texts_features, val_texts_features, test_texts_features
    
def basic_model(X, X_test, Y, Y_test, flag = 'svc'):
    if flag == 'linear svc':
        model = sklearn.svm.LinearSVC()
    elif flag == 'logistic':
        model = sklearn.linear_model.LogisticRegression(n_jobs = -1, max_iter = 1000)
    elif flag == 'naive bayes':
        model = sklearn.naive_bayes.MultinomialNB()
    elif flag == 'svc':
        model = sklearn.svm.SVC(probability = True)
        
    model.fit(X, Y)
    Y_test_hat = model.predict(X_test)
    report = sklearn.metrics.classification_report(Y_test, Y_test_hat, digits = 3)
    accuracy = sklearn.metrics.accuracy_score(Y_test, Y_test_hat)
    auc_score = sklearn.metrics.roc_auc_score(Y_test, Y_test_hat)
    fp_rate, tp_rate, thresholds = sklearn.metrics.roc_curve(Y_test, Y_test_hat)
    
    print("report {}".format(report))
    print('accuracy score {:.4f}'.format(accuracy))
    print('auc_score {:.4f}'.format(auc_score))
    
    plt.plot(fp_rate, tp_rate)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show()
    
    return accuracy, auc_score
    
def main():
    
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_file_path', type = str, help = 'Path to the data file')
    parser.add_argument('--ngram', type = int, help = 'Define the max ngram', default = 3)
    parser.add_argument('--model_name', type = str, help = 'model name')
    
    args = parser.parse_args()
    
    print('Reading in {}'.format(args.input_file_path.split('/')[-1]))
    train_text, train_target, val_text, val_target, test_text, test_target = pd.read_pickle(args.input_file_path)
    accuracies = {}
    aucs = {}
    
    train_features, val_features, test_features = generate_features(train_text, val_text, test_text, max_ngram = args.ngram)
    
    conditions = ['cohort', #0
                  'Obesity', #1
                  'Non_Adherence', #2
                  'Developmental_Delay_Retardation', #3
                  'Advanced_Heart_Disease', #4
                  'Advanced_Lung_Disease', #5
                  'Schizophrenia_and_other_Psychiatric_Disorders', #6
                  'Alcohol_Abuse', #7
                  'Other_Substance_Abuse', #8
                  'Chronic_Pain_Fibromyalgia', #9
                  'Chronic_Neurological_Dystrophies', #10
                  'Advanced_Cancer', #11
                  'Depression', #12
                  'Dementia', #13
                  'Unsure'] #14
    
    for i, j in enumerate(conditions):
        print('Now working on condition {}'.format(j))
        train_y = train_target[:, i]
        test_y = test_target[:, i]
        accuracy, auc_score = basic_model(train_features, test_features, train_y, test_y, flag = args.model_name)
        accuracies[j] = accuracy
        aucs[j] = auc_score
    
    print('accuracy: {}'.format(accuracies))
    print('auc: {}'.format(aucs))
    
if __name__ == '__main__':
    main()