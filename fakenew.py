import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# loading the model
with open('fake_new_log.pkl', 'rb') as fakenews:
    model = pickle.load(fakenews)

# loading the data
df = pd.read_csv('News.csv', index_col=0)
data = df.drop(["title", "subject", "date"], axis=1)
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)

vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)
model = LogisticRegression()
# fitting the model on training data
model.fit(x_train, y_train)

def News(text):
    # Transform the input text using the same vectorizer and transformer
    vectorized_text = vectorization.transform([text])
    # Make predictions with the trained model
    results = model.predict(vectorized_text)
    return results

def main():
    st.title("Fake News Prediction")
    text = st.text_input("Enter a News:")
     # Add a button to trigger predictions
    if st.button("predict"):
        if text:
            prediction = News(text)
            st.text("Predictions:")
            st.text(prediction)

if __name__ == '__main__':
    main()
