import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommenderSystem:

        def __init__(self, song_name: str, artist_name: str, number_of_recommendations: int, weight_content_based: float,
                    weight_collaborative :float,songs_data,transformed_matrix,interaction_matrix,track_ids):
            
            self.number_of_recommendations = number_of_recommendations
            self.song_name = song_name.strip().lower()
            self.artist_name = artist_name.strip().lower()
            self.weight_content_based =weight_content_based
            self.weight_collaborative =weight_collaborative
            self.songs_data =songs_data
            self.transformed_matrix =transformed_matrix
            self.interaction_matrix =interaction_matrix
            self.track_ids =track_ids


        def calculate_content_based_similarities(self,song_name,artist_name,songs_data,transformed_matrix):
            # filter out the songs from the data
            song_row = songs_data.loc[(songs_data["name"].str.strip().str.lower()==song_name) & (songs_data["artist"].str.strip().str.lower()==artist_name)]
            
            # if no row found
            if song_row.empty:
                raise ValueError(f"Song not found: {song_name} by {artist_name}")
            
            # get the index of the song
            song_index = song_row.index[0]

            # generate the input vector
            input_vector = transformed_matrix[song_index].reshape(1,-1)

            # calculate similarity score
            content_similarity_scores = cosine_similarity(input_vector,transformed_matrix)

            return content_similarity_scores
        

        def calculate_collaborative_filtering_similarities(self,song_name,artist_name,track_ids,songs_data,interaction_matrix):
            
            # fetch the row from songs data
            song_row = songs_data.loc[(songs_data["name"].str.strip().str.lower()==song_name) & (songs_data["artist"].str.strip().str.lower()==artist_name)]
            
            # if no row found
            if song_row.empty:
                raise ValueError(f"Song not found: {song_name} by {artist_name}")
            # track id of input row
            input_track_id = song_row['track_id'].values.item()

            # index value of track_id
            ind = np.where(track_ids==input_track_id)[0].item()

            # fetch the input vector
            input_vector = interaction_matrix[ind].reshape(1,-1)

            # get similarity scores
            collaborative_similarity_scores = cosine_similarity(input_vector,interaction_matrix)

            return collaborative_similarity_scores
        

        def normalize_similarities(self,similarity_scores):
            minimum = np.min(similarity_scores)
            maximum = np.max(similarity_scores)

            if maximum == minimum:
                return np.zeros_like(similarity_scores)
            normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
            return normalized_scores
        
        
        def weighted_combination(self,content_based_scored,collaborative_filtering_scores):
            weighted_scores = (self.weight_content_based * content_based_scored) + (self.weight_collaborative * collaborative_filtering_scores)
            return weighted_scores
        


        def give_recommendations(self):
            # calculate content based similarity score 
            content_based_similarities = self.calculate_content_based_similarities(song_name= self.song_name,
                                                                                 artist_name = self.artist_name,
                                                                                 songs_data = self.songs_data,
                                                                                 transformed_matrix = self.transformed_matrix)
            
            # calculate collaborative similarity score
            collaborative_filtering_similarities = self.calculate_collaborative_filtering_similarities(song_name = self.song_name,
                                                                                                       artist_name = self.artist_name,
                                                                                                       track_ids = self.track_ids,
                                                                                                       songs_data=self.songs_data,
                                                                                                       interaction_matrix = self.interaction_matrix )

            # normalizing the scores
            normalized_content_based_similarities = self.normalize_similarities(content_based_similarities)
            normalized_collaborative_filtering_similarities = self.normalize_similarities(collaborative_filtering_similarities)

            # weighted combination of similarities
            weighted_score = self.weighted_combination(content_based_scored = normalized_content_based_similarities,
                                                       collaborative_filtering_scores = normalized_collaborative_filtering_similarities)
            
            # index value of recommendation
            recommendation_indices = np.argsort(weighted_score.ravel())[-self.number_of_recommendations-1:][::-1]

            # get top k recommendations
            recommendation_track_ids = self.track_ids[recommendation_indices]

            top_k_songs = (
                    self.songs_data[self.songs_data['track_id'].isin(recommendation_track_ids)]
                    .set_index('track_id')
                    .reindex(recommendation_track_ids)
                    .reset_index()
                )
            return top_k_songs 
        


if __name__ == "__main__":
    # load the transformed data
    transformed_data = load_npz('data/processed/transformed_hybrid_data.npz')

    # load the interaction matrix
    interaction_matrix = load_npz('data/processed/interaction_matrix.npz')

    # load the track ids
    track_ids = np.load('data/processed/track_ids.npy',allow_pickle=True)

    # load the songs data
    songs_data = pd.read_csv("data/processed/collab_filtered_data.csv",usecols=["track_id","name","artist","spotify_preview_url"])

    # create an instance of HybricRecommenderSystem
    song_name = "hips don't lie"
    artist_name = "shakira"

    hybrid_recommender = HybridRecommenderSystem(song_name = song_name,
                                                 artist_name = artist_name,
                                                 number_of_recommendations= 10, 
                                                 weight_content_based= 0.3,
                                                 weight_collaborative =0.7,
                                                 songs_data = songs_data,
                                                 transformed_matrix = transformed_data,
                                                 interaction_matrix = interaction_matrix,
                                                 track_ids = track_ids)
    

    # get recommendation
    recommendations = hybrid_recommender.give_recommendations()

    print(recommendations)