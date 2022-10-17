# core packages
import pickle
import numpy as np
import altair as alt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
matplotlib.use('Agg')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# class_names

class_names = ['Fake', 'Real']

# Main application


def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        with st.form(key="mlform"):
            col1, col2 = st.columns([2, 1])
            with col1:
                message = st.text_area("Message")
                submit_message = st.form_submit_button(label="predict")

            with col2:
                st.write("Online ML")
                st.write("Predict Texts as Fake or Real News")
        if submit_message:
            prediction = model.predict([message])
            prediction_proba = model.predict_proba([message])
            probability = max(model.predict_proba([message]))

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("Original Text")
                st.write(message)

                st.success("Prediction")
                st.write(prediction)

            with res_col2:
                st.info("Probability")
                st.write(prediction_proba)

                # plot of the probability
                df_proba = pd.DataFrame({'label': class_names,
                                         'probability': (np.reshape(pd.DataFrame(prediction_proba).values, -1))})
                # st.dataframe(df_proba)
                # visualization
                df_proba.columns = ['label', 'probability']
                fig = alt.Chart(df_proba).mark_bar().encode(
                    x='label',
                    y='probability')
                st.altair_chart(fig, use_container_width=True)

    else:
        st.subheader("About")
        st.write("This is an web app used to classify if a text is fake or real news")


if __name__ == "__main__":
    main()
