import pickle
import streamlit as st

model = pickle.load(open('model/model_for_fakeNews (1).pkl', 'rb'))
vector = pickle.load(open('model/vectors (1).pkl', 'rb'))

st.header("Fake News Detection")

news = st.text_area(label='Enter news here')

if st.button('Predict'):
    result = model.predict(vector.transform([news]).toarray())
    if result == 1:
        st.header("This news is REAL")
    else:
        st.header("This news is FAKE")


