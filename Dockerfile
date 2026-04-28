# set up the base image
FROM python:3.13.5 

# set the working directory
WORKDIR /app/

# copy requirements.txt file to workdir
COPY requirements.txt .

# install requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy the nearest neighbor model
COPY ./models/nearest_neighbor_cbf.joblib ./models/
# copy the data files
COPY ./data/processed/cleaned_music_data.csv \
    ./data/processed/collab_filtered_data.csv \
    ./data/processed/interaction_matrix.npz \
    ./data/processed/track_ids.npy \
    ./data/processed/transformed_data.npz \
    ./data/processed/transformed_hybrid_data.npz \
    ./data/processed/

# copy the code files
COPY ./app.py .

COPY ./src/data_cleaning.py \
    ./src/transform_filtered_data.py \
    ./src/collaborative_filtering.py \
    ./src/content_based_filtering.py \
    ./src/hybrid_recommendation.py \
    ./src/

# expose the port
EXPOSE 8000

# run streamlit app
CMD [ "streamlit","run","app.py","--server.port","8000" ]