import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm

parameter_space = {
    'hidden_layer_sizes' : [(100,), (100, 100), (50, 50), (64, 32)]
}
mlp = MLPClassifier(solver="lbfgs", alpha=0.005, hidden_layer_sizes=(100,))
clf = GridSearchCV(mlp, parameter_space, n_jobs=4, cv=3)

dataSet = pd.read_csv("data/train/dataSetNonBinary.csv")
dataSet.dropna(inplace=True)
corpus = dataSet["content"].tolist()
vectorizer = CountVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
dense_array = X.toarray()
df = pd.DataFrame(dense_array, columns=feature_names)

y = dataSet["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)

clf.fit(X_train, Y_train)
print('Best parameters found:\n', clf.best_params_)