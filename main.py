import unfairClauseDetectionModel, unfairClauseClassification
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load



def mainMenu():
    try:
        classifier = load("model/unfairClauseDetectionModel.joblib")
        load("model/vectorizer.joblib")
        print("\n#####   main menu   #####\ncurrent model: " + str(type(classifier)) + "\n\n1: choose Identification Model\n2: detect potentialy unfair clauses in file\n3: classify clauses in file")
    except:
        print("\n#####   main menu   #####\ncurrent model: " + "none" + "\n\n1: choose Identification Model\n2: detect potentialy unfair clauses in file\n3: classify clauses in file")
    
    userInput = input("Enter something: ")
    print()
    
    if userInput == "1":
        return chooseModelMenu()
    
    elif userInput == "2":
        try:
            load("model/unfairClauseDetectionModel.joblib")
            load("model/vectorizer.joblib")
        except :
            print("no model detected: please load a model from the menu")
            return chooseModelMenu()
        userInput = input("enter file path: ")
        return unfairClauseDetectionModel.detectUnfairClauseInTextToString(userInput)
    
    elif userInput == "3":
        userInput = input("enter file path: ")
        try:
            load("model/unfairClauseClassificationModel.joblib")
            load("model/unfairClauseClassificationModel.joblib")
            print("begining classification")
        except:
            print("no model detected, training Model")
            unfairClauseClassification.trainModel()
            print("finished Training Model, begining classification")

        return unfairClauseClassification.detectUnfairClauseInTextToString(userInput)
    
    else :
        print("error unrecognised input")
        return mainMenu()
    
def chooseModelMenu():
    print("\n#####   choose Identification Model Menu   #####\n\n0: go to main menu\n1: MultinomialNB\n2: ComplementNB\n3: GaussianNB\n4: SVC\n")
    userInput = input("Enter something: ")
    print()
    
    if userInput == "0":
        return mainMenu()
    
    elif userInput == "1":
        print("training MultinomialNBModel")
        unfairClauseDetectionModel.trainMultinomialNBModel()
        return mainMenu()
    
    elif userInput == "2" :
        print("training ComplementNBModel")
        unfairClauseDetectionModel.trainComplementNBModel()
        return mainMenu()
    
    elif userInput == "3" :
        print("training GaussianNBModel")
        unfairClauseDetectionModel.trainGaussianNBModel()
        return mainMenu()
    
    elif userInput == "4" :
        print("training SVCModel")
        unfairClauseDetectionModel.trainSVCModel()
        return mainMenu()
    
    else :
        print("error unrecognised input")
        return chooseModelMenu()

if __name__ == "__main__" :
    mainMenu()
    