# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 
import requests

# EDA Pkgs
import pandas as pd 
import numpy as np 

import joblib 
pipe_lr = joblib.load(open("models/ecommerce_model.pkl","rb"))



#Function
def predict_ecommerce(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

# emotions_emoji_dict = {"Household","Books", "Electronics", "Clothing & Accessories"}


# Main Application
def main():
	st.title("Ecommerce Classifier App")
	menu = ["Home","Monitor","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home-Ecommerce In Text")

		with st.form(key='ecommerce_clf_form'):
			raw_text = st.text_area("ecommerce Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Function Here
			prediction = predict_ecommerce(raw_text)
			probability = get_prediction_proba(raw_text)
			
			#add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				st.write(prediction)
                


			with col2:
				st.success("Prediction Probability")
				#st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				#st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["ecommerce","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='ecommerce',y='probability',color='ecommerce')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "Monitor":
		st.subheader("Monitor App")

		


	else:
		st.subheader("About")





if __name__ == '__main__':
	main()