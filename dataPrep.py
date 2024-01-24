import pandas as pd
import sklearn as sk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
import string

def tokenisationAndNormalisation(str:str):
    #normalisation
    str = [char.lower() for char in str if char.lower() not in string.punctuation]
    str = ''.join(str)
    
    #tokenisation en enlevant les mots peu important (stopwords)
    return [word for word in str.split() if word not in stopwords.words("english")]

def prepareBinaryDataSet():
    #dataset recuperation
    dataSet = pd.read_csv("data/train/train_extract_20200101_000000.csv")
    print(dataSet.shape)

    #enlever les colonnes inutiles pour en garder seulement 2
    dataSet.drop(["line_number", "fairness", "category"], axis=1, inplace=True)
    dataSet.drop_duplicates(inplace=True)


    #encodage des labels
    mapOfLabel = {'a2': 1, "a3": 1, "a4": 1, "ch2": 1, "ch3": 1, "cr2": 1, "cr3": 1, "j1": 0, "j2": 1, "j3": 1, "law1": 0, "law2": 1, "ltd1": 0, "ltd2": 1, "ltd3": 1, "ter3": 1, "unc": 0, "use2": 1}
    dataSet["label"] = dataSet["label"].map(mapOfLabel)
    dataSet["label"] = dataSet["label"].convert_dtypes(int) #pour enlever les flotants

    #tokenisation et normalisation pour préparer les unigrammes et bigrammes
    dataSet["content"] = dataSet["content"].apply(tokenisationAndNormalisation)

    #save
    dataSet.to_csv("data/train/dataSet.csv")
    print("dataset saved in data/train/dataSet.csv")

def prepareNonBinaryDataSet():
    #dataset recuperation
    dataSet = pd.read_csv("data/train/train_extract_20200101_000000.csv")
    print(dataSet.shape)

    #enlever les colonnes inutiles pour en garder seulement 2
    dataSet.drop(["line_number", "fairness", "category"], axis=1, inplace=True)
    dataSet.drop_duplicates(inplace=True)


    #encodage des labels
    mapOfLabel = {'a2': 1, "a3": 1, "a4": 1, "ch2": 2, "ch3": 2, "cr2": 3, "cr3": 3, "j1": 4, "j2": 4, "j3": 4, "law1": 5, "law2": 5, "ltd1": 6, "ltd2": 6, "ltd3": 6, "ter3": 7, "unc": 8, "use2": 9}
    dataSet["label"] = dataSet["label"].map(mapOfLabel)
    dataSet["label"] = dataSet["label"].convert_dtypes(int) #pour enlever les flotants

    #tokenisation et normalisation pour préparer les unigrammes et bigrammes
    dataSet["content"] = dataSet["content"].apply(tokenisationAndNormalisation)

    #save
    dataSet.to_csv("data/train/dataSetNonBinary.csv")
    print("dataset saved in data/train/dataSetNonBinary.csv")

if __name__ == "__main__" :
    prepareBinaryDataSet()
    prepareNonBinaryDataSet()
