import streamlit as st
from src.content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd
import joblib

# transformed data path
transformed_data_path = "data/processed/transformed_data.npz"

# cleaned data path
cleaned_data_path = "data/processed/cleaned_music_data.csv"

# trained model path
model_path = 'models/nearest_neighbor.joblib'

# using streamlit caching to save rerun
@st.cache_data
def load_dataframe(cleaned_data_path):
    # load the data
    data = pd.read_csv(cleaned_data_path)
    song_to_index = {
        name.lower() : idx for idx,name in enumerate(data["name"])
    }
    return data,song_to_index


@st.cache_data
def load_transformed_data(transformed_data_path):
    # load the transformed data
    transformed_data = load_npz(transformed_data_path)
    return transformed_data



@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model 


data,song_to_index = load_dataframe(cleaned_data_path)
transformed_data = load_transformed_data(transformed_data_path)
model = load_model(model_path)


# title
st.title("Welcome to Spotify Song Recommender ! ")

# subheader
st.write("### Enter the name of a song and the recommender will suggest similar songs ")

# text input 
song_name = st.text_input("Enter a song name : " )
st.write("You Entered : ", song_name)

# k recommendation
k = st.selectbox("How many recommendations do you want ? ",[5,10,15,20],index=1)

# button 
try :
    if st.button("Get Recommendation"):
        st.write("Recommendation for ",f"***{song_name}***")
        recommendation = recommend(song_name,data,model,song_to_index,transformed_data,k)

        # display recommendation
        for ind,recommendation in recommendation.iterrows():
            song_name = recommendation['name'].title()
            artist_name = recommendation['artist'].title()

            if ind == 0:
                st.markdown("## Currently Playing")
                st.markdown(f"### **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('----')

            elif ind == 1:
                st.markdown("## Next Up")
                st.markdown(f"### **{ind}.** **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('----')
            else :
                st.markdown(f"### **{ind}.** **{song_name}** by **{artist_name}**")
                st.audio(recommendation['spotify_preview_url'])
                st.write('----')

except Exception as e :
    print(e)
    st.markdown("### Oops ! Can not find song in my Data")
    st.markdown("### Please try another song.")
