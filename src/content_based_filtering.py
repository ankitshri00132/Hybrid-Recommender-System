import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz

from src.data_cleaning import data_for_content_filtering

# cleaned data path
CLEANED_DATA_PATH = "data/processed/cleaned_music_data.csv"

# cols to transform 
frequence_encode_cols = ['year']
ohe_cols = ['artist','time_signature','key']
tfidf_col = 'tags'
standard_scale_cols = ['duration_ms','loudness','tempo']
min_max_scale_cols = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']


def train_transformer(data):
    """
    Trains a ColumnTransformer on the provided data and saves the transformer to a file.
    The ColumnTransformer applies the following transformations:
    - Frequency Encoding using CountEncoder on specified columns.
    - One-Hot Encoding using OneHotEncoder on specified columns.
    - TF-IDF Vectorization using TfidfVectorizer on a specified column.
    - Standard Scaling using StandardScaler on specified columns.
    - Min-Max Scaling using MinMaxScaler on specified columns.
    Parameters:
    data (pd.DataFrame): The input data to be transformed.
    Returns:
    None
    Saves:
    transformer.joblib: The trained ColumnTransformer object.
    """

    # transformer
    transformer = ColumnTransformer(
        transformers = [
            ("frequency_encode",CountEncoder(normalize=True,return_df=True),frequence_encode_cols),
            ("ohe",OneHotEncoder(handle_unknown='ignore'),ohe_cols),
            ("tfidf",TfidfVectorizer(max_features=85),tfidf_col),
            ("standard_scale",StandardScaler(),standard_scale_cols),
            ("min_max_scale",MinMaxScaler(),min_max_scale_cols)
        ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False
        )
    
    # fit the transformer
    transformer.fit(data)

    # save the transformer
    joblib.dump(transformer , "transformer.joblib")


def transform_data(data):

    """
    Transforms the input data using a pre-trained transformer.
    Args:
        data (array-like): The data to be transformed.
    Returns:
        array-like: The transformed data.
    """

    # load the transformer
    transformer = joblib.load("transformer.joblib")

    # transform the data
    transformed_data  = transformer.transform(data)

    return transformed_data


def save_transformed_data (tansformed_data,save_path):

    """
    Save the transformed data to a specified file path.

    Parameters:
    transformed_data (scipy.sparse.csr_matrix): The transformed data to be saved.
    save_path (str): The file path where the transformed data will be saved.

    Returns:
    None
    """
 
    # save the transformed data
    save_npz(save_path,tansformed_data)



def build_model(transformed_data):
    """
    Builds a nearest neightbor model using sparse matrix
    parameter: transformed_data (sparse matrix)

    returns : Trained NN Model 
    """
    model = NearestNeighbors(metric='cosine',algorithm='brute')
    model.fit(transformed_data)
    return model


def save_model(model):
    """
    Saves the model into the "models" folder
    parameter:
    model :  A trained model
    """
    joblib.dump(model,"models/nearest_neighbor.joblib")



def recommend(song_name ,songs_data,model,song_to_index,transformed_data,k=10) :
    """
        Recommends top k songs similar to the given song based on content-based filtering.

        Parameters:
        song_name (str): The name of the song to base the recommendations on.
        songs_data (DataFrame): The DataFrame containing song information.
        model : Trained Nearest Neighbour Model
        songs_to_index : A precomputed dictionary of all the songs with their indexes
        transformed_data (sparse matrix): The transformed data matrix for similarity calculations.
        k (int, optional): The number of similar songs to recommend. Default is 10.

        Returns:
        DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """
    try : 
        # convert song name to lower case
        song_name = song_name.lower()

        # get the index of the songs
        song_idx = song_to_index.get(song_name)

        if song_idx is None:
            raise AttributeError("Not found in our data")
        else:
            # generate input vector
            input_vector = transformed_data[song_idx].reshape(1,-1)

            # find nearest neighbours -> returns indexes of K neighbours
            distances, indices = model.kneighbors(input_vector,n_neighbors=k+1)

            # top k songs' index
            indices = indices.ravel()

            # return the top k songs
            top_k_list =(
                        songs_data.iloc[indices]
                        [["name","artist","spotify_preview_url"]]
                        .reset_index(drop=True)
                        )

            return top_k_list
    except Exception as e:
        print("Error occurred : ",e)
        



def main(data_path):
    """
    Main function executing every other functions 

    Parameters:
    data_path (str): The path to the CSV file containing the song data.

    Returns: None
    """
    # load the data
    data = pd.read_csv(data_path)

    # clean the data
    data_content_filtering = data_for_content_filtering(data)

    # train the transformer
    train_transformer(data_content_filtering)

    # transform the data
    transformed_data = transform_data(data_content_filtering)

    # save transformed data
    save_transformed_data(transformed_data, "data/processed/transformed_data.npz")

    # build model 
    model = build_model(transformed_data)

    # save the model
    save_model(model)
    

if __name__=="__main__":
    main(CLEANED_DATA_PATH)