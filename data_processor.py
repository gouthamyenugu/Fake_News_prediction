import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import  train_test_split

# Read data from path
def prepare_data(path_to_data):
    data = pd.read_csv(path_to_data)
    X = data['text']
    y = data['class']
    return {'text' : X,'class' : y}


def create_train_test_data(X,y,test_size,random_state):
    cv = TfidfVectorizer()
    X = cv.fit_transform(X)
    X_train, X_test, y_train,y_test = train_test_split(X,y,
                                                       test_size =test_size,random_state = random_state)
    return {'X_train':X_train,'X_test':X_test,
           'y_train':y_train,'y_test':y_test},cv
