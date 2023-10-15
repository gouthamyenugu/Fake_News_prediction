from streamlit import success
import streamlit as st
import joblib

# Load the model
news_log = joblib.load(open('my_fake_news.pkl', 'rb'))
# Load Vectorizer
vectorizer = joblib.load(open('my_vectorizer_news.pkl', 'rb'))
# Main function
def main():
    st.title("Fake News Detection System")
    text_message = st.text_input("Enter your News")
    if st.button("Detect"):
        if text_message:
            prediction = news_log.predict(vectorizer.transform([text_message]))
                       
            if prediction[0] == 1:
                info = "Fake News"
            else:
                info = "Not a Fake News"
            st.success('News: {}'.format(info))

# Run the Streamlit app
if __name__ == '__main__':
    main()
