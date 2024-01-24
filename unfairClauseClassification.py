import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import nltk
import string

def trainModel():

    dataSet = pd.read_csv("dataSetNonBinary.csv")
    dataSet.dropna(inplace=True)
    corpus = dataSet["content"].tolist()
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    dense_array = X.toarray()
    df = pd.DataFrame(dense_array, columns=feature_names)

    y = dataSet["label"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    #model = svm.LinearSVC(dual="auto", C=0.078)
    model = MLPClassifier(solver="lbfgs", alpha=0.005, hidden_layer_sizes=(100,))

    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    print(classification_report(Y_test, prediction))
    print(confusion_matrix(Y_test, prediction))

    dump(vectorizer, "vectorizerClauseClassification.joblib")
    dump(model, "unfairClauseClassificationModel.joblib")
    
def detectUnfairClauseInText(path:str):
    def detectIfUnfairClauseInLine(line:str):
        x = vectorizer.transform(line).toarray()
        pred = classifier.predict(x)
        return int(str(pred[0])[:1]) #mise sous forme d'entier car pred est de type [float] et de taille 1

    try:
        file = open(path, 'r')
        text = file.read()
        file.close()
    except:
        print("file not found: " + path)
        return 0
        
    vectorizer:CountVectorizer = load("vectorizerClauseClassification.joblib")
    classifier:MLPClassifier = load("unfairClauseClassificationModel.joblib")

    dict = {"a":[], "ch":[], "cr":[], "j":[], "law":[], "ltd":[], "ter": [], "unc": [], "use":[]}
    swichDict = {1:"a", 2:"ch", 3: "cr", 4:"j", 5:"law", 6:"ltd", 7:"ter", 8:"unc", 9:"use"}
    
    text = nltk.sent_tokenize(text) #sentence by sentence
    for sentence in text:
        detectionRes = detectIfUnfairClauseInLine([sentence.translate(str.maketrans('', '', string.punctuation))])
        dict.get(swichDict.get(detectionRes)).append(sentence)  
    return dict

def detectUnfairClauseInTextToString(path:str):
    res = detectUnfairClauseInText(path)
    for key, value in res.items():
        print(key)
        if(key != "unc"):
            for str in value:
                print(" "+str)
        print()
        
