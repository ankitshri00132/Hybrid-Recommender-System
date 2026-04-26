import streamlit as st
from scipy.sparse import load_npz
import pandas as pd
import joblib
from numpy import load
from src.content_based_filtering import content_recommendation
from src.collaborative_filtering import collaborative_recommendation
from src.hybrid_recommendation import HybridRecommenderSystem as hrs


# transformed data path
transformed_data_path = "data/processed/transformed_data.npz"
# cleaned data path
cleaned_data_path = "data/processed/cleaned_music_data.csv"
# trained model path
cbf_model_path = 'models/nearest_neighbor_cbf.joblib'
# filtered songs data path
filtered_songs_data_path ='data/processed/collab_filtered_data.csv'
# transformed hybrid data used in hybrid recommender
transformed_hybrid_data_path = 'data/processed/transformed_hybrid_data.npz'
# track_ids path
track_ids_path = 'data/processed/track_ids.npy'
# interaction matrix path
interaction_matrix_path = 'data/processed/interaction_matrix.npz'



# using streamlit caching to save rerun
@st.cache_data
def load_dataframe(cleaned_data_path):
    # load the data
    data = pd.read_csv(cleaned_data_path)
    song_artist_index = {
       (name.lower(),artist.lower()) : index for index , (name,artist) in enumerate(zip(data['name'],data['artist']))
    }
    return data,song_artist_index


@st.cache_data
def load_transformed_data(transformed_data_path):
    transformed_data = load_npz(transformed_data_path)
    return transformed_data


@st.cache_data
def load_transformed_hybrid_data(transformed_hybrid_data_path):
    transformed_data = load_npz(transformed_hybrid_data_path)
    return transformed_data


@st.cache_data
def load_filtered_data(filtered_songs_data_path):
    # load the filtered data
    filtered_data = pd.read_csv(filtered_songs_data_path)
    return filtered_data


@st.cache_data
def load_track_ids(track_ids_path):
    # load the track ids
    track_ids = load(track_ids_path,allow_pickle=True)
    return track_ids


@st.cache_data
def load_interaction_matrix(interaction_matrix_path):
    # load the interaction matrix
    interaction_matrix = load_npz(interaction_matrix_path)
    return interaction_matrix


@st.cache_resource
def load_model(model_path):
    # load the nearest neighbor model for content based filtering
    model = joblib.load(model_path)
    return model 


# laoding all the data and files
data,song_artist_index = load_dataframe(cleaned_data_path)
transformed_data = load_transformed_data(transformed_data_path)
cbf_model = load_model(cbf_model_path)
filtered_data = load_filtered_data(filtered_songs_data_path)
transformed_hybrid_data = load_transformed_hybrid_data(transformed_hybrid_data_path)
track_ids = load_track_ids(track_ids_path)
interaction_matrix = load_interaction_matrix(interaction_matrix_path)


# title
st.title("Welcome to Spotify Song Recommender ! ")

# subheader
st.write("### Enter the name of a song and artist and the recommender will suggest similar songs ")

# text input 
song_name = st.text_input("Enter a song name : " )
st.write("You Entered : ", song_name)
song_name = song_name.strip()

# artist input 
artist_name = st.text_input("Enter the artist name : " )
st.write("You Entered : ", artist_name)
artist_name = artist_name.strip()


# type of filtering
filtering_type = st.selectbox("Select the type of filtering : ",["Content-Based Filtering","Collaborative Filtering","Hybrid Recommender System"],index=2)
# k recommendation
k = st.selectbox("How many recommendations do you want ? ",[5,10,15,20],index=1)

# button 
try :
    if filtering_type == 'Content-Based Filtering':
        if st.button("Get Recommendation"):
            st.write("Recommendation for ",f" **{song_name}** by **{artist_name}**")
            recommendations = content_recommendation(song_name,artist_name,data,cbf_model,song_artist_index,transformed_data,k)

            # display recommendation
            for ind,recommendation in recommendations.iterrows():
                rec_song = recommendation['name'].title()
                rec_artist = recommendation['artist'].title()

                if ind == 0:
                    st.markdown("## Currently Playing")
                    st.markdown(f"### **{rec_song}** by **{rec_artist}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('----')

                elif ind == 1:
                    st.markdown("## Next Up")
                    st.markdown(f"### **{ind}.** **{rec_song}** by **{rec_artist}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('----')
                else :
                    st.markdown(f"### **{ind}.** **{rec_song}** by **{rec_artist}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('----')

    elif filtering_type == 'Collaborative Filtering':
         if st.button("Get Recommendation"):
            if((filtered_data['name'].str.lower() == song_name.lower()) & (filtered_data['artist'].str.lower() == artist_name.lower())).any():
                st.write("Recommendation for ",f" **{song_name}** by **{artist_name}**")
                
                recommendations = collaborative_recommendation(
                                                            song_name=song_name,
                                                            artist_name=artist_name,
                                                            track_ids=track_ids,
                                                            songs_data=filtered_data,
                                                            interaction_matrix=interaction_matrix,
                                                            k=k
                                                            )

                # display recommendation
                for ind,recommendation in recommendations.iterrows():
                    rec_song = recommendation['name'].title()
                    rec_artist = recommendation['artist'].title()
                    if ind == 0:
                        st.markdown("## Currently Playing")
                        st.markdown(f"### **{rec_song}** by **{rec_artist}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('----')

                    elif ind == 1:
                        st.markdown("## Next Up")
                        st.markdown(f"### **{ind}.** **{rec_song}** by **{rec_artist}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('----')
                    else :
                        st.markdown(f"### **{ind}.** **{rec_song}** by **{rec_artist}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('----')
            
    elif filtering_type == 'Hybrid Recommender System':
         if st.button("Get Recommendation"):
            if((filtered_data['name'].str.lower() == song_name.lower()) & (filtered_data['artist'].str.lower() == artist_name.lower())).any():
                st.write("Recommendation for ",f" **{song_name}** by **{artist_name}**")
                
                recommender = hrs(song_name = song_name,
                                                 artist_name = artist_name,
                                                 number_of_recommendations= k, 
                                                 weight_content_based= 0.2,
                                                 weight_collaborative =0.9,
                                                 songs_data = filtered_data,
                                                 transformed_matrix = transformed_hybrid_data,
                                                 interaction_matrix = interaction_matrix,
                                                 track_ids = track_ids)
                recommendations = recommender.give_recommendations()
                # display recommendation
                for ind,recommendation in recommendations.iterrows():
                    rec_song = recommendation['name'].title()
                    rec_artist = recommendation['artist'].title()
                    if ind == 0:
                        st.markdown("## Currently Playing")
                        st.markdown(f"### **{rec_song}** by **{rec_artist}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('----')

                    elif ind == 1:
                        st.markdown("## Next Up")
                        st.markdown(f"### **{ind}.** **{rec_song}** by **{rec_artist}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('----')
                    else :
                        st.markdown(f"### **{ind}.** **{rec_song}** by **{rec_artist}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('----')
            

except Exception as e :
    print(e)
    st.markdown("### Oops ! Can not find song in my Data")
    st.markdown("### Please try another song.")
