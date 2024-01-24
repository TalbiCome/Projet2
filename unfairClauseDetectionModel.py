import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from sklearn import svm

def trainMultinomialNBModel():
    dataSet = pd.read_csv("dataSet.csv")

    dataSet = dataSet.dropna()
    x = dataSet["content"]
    y = dataSet["label"]

    #generation de bigramme
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
    x = vectorizer.fit_transform(x)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    classifier = MultinomialNB()

    classifier = MultinomialNB().fit(X_train, Y_train)
    prediction = classifier.predict(X_test)
    print(classification_report(Y_test, prediction))

    dump(vectorizer, "vectorizer.joblib")
    dump(classifier, "unfairClauseDetectionModel.joblib")
    print("model saved in unfairClauseDetectionModel.joblib")
    
def trainComplementNBModel():
    dataSet = pd.read_csv("dataSet.csv")

    dataSet = dataSet.dropna()
    x = dataSet["content"]
    y = dataSet["label"]

    #generation de bigramme
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
    x = vectorizer.fit_transform(x)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    classifier = ComplementNB()

    classifier = ComplementNB().fit(X_train, Y_train)
    prediction = classifier.predict(X_test)
    print(classification_report(Y_test, prediction))

    dump(vectorizer, "vectorizer.joblib")
    dump(classifier, "unfairClauseDetectionModel.joblib")
    print("model saved in unfairClauseDetectionModel.joblib")

def trainGaussianNBModel():
    dataSet = pd.read_csv("dataSet.csv")

    dataSet = dataSet.dropna()
    x = dataSet["content"]
    y = dataSet["label"]

    #generation de bigramme
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
    x = vectorizer.fit_transform(x)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()

    classifier = GaussianNB().fit(X_train.toarray(), Y_train)
    prediction = classifier.predict(X_test.toarray())
    print(classification_report(Y_test, prediction))

    dump(vectorizer, "vectorizer.joblib")
    dump(classifier, "unfairClauseDetectionModel.joblib")
    print("model saved in unfairClauseDetectionModel.joblib")

def trainSVCModel():
    dataSet = pd.read_csv("dataSet.csv")
    dataSet.dropna(inplace=True)
    corpus = dataSet["content"].tolist()
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    dense_array = X.toarray()
    df = pd.DataFrame(dense_array, columns=feature_names)

    y = dataSet["label"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    model = svm.SVC()

    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    print(classification_report(Y_test, prediction))
    print(confusion_matrix(Y_test, prediction))

    dump(vectorizer, "vectorizer.joblib")
    dump(model, "unfairClauseDetectionModel.joblib")


def detectUnfairClauseInText(path:str):
    def detectIfUnfairClauseInLine(line:str):
        x = vectorizer.transform(line).toarray()
        pred = classifier.predict(x)
        return int(str(pred[0])[:1]) #mise sous forme d'entier car pred est de type [float] et de taille 1
    
    try:
        file = open(path, 'r', encoding="utf-8")
        text = file.read()
        file.close()
    except:
        print("file not found: " + path)
        return 0
     
    vectorizer:CountVectorizer = load("vectorizer.joblib")
    classifier:MultinomialNB = load("unfairClauseDetectionModel.joblib")
    
    res = []
    text = nltk.sent_tokenize(text) #sentence by sentence
    for sentence in text:
        detectionRes = detectIfUnfairClauseInLine([sentence.translate(str.maketrans('', '', string.punctuation))])
        if(detectionRes):
            print(detectionRes)
            res.append([sentence])
        
    return res

def detectUnfairClauseInTextToString(path:str):
    res = detectUnfairClauseInText(path)
    for line in res:
        print(line)