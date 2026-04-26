import pandas as pd
from src.data_cleaning import data_for_content_filtering
from src.content_based_filtering import transform_data,save_transformed_data

# path of filtered data
FILTERED_DATA_PATH = 'data/processed/collab_filtered_data.csv'

# save file path
SAVE_PATH = 'data/processed/transformed_hybrid_data.npz'

def main(data_path,save_path):
    # load the filtered data
    filtered_data = pd.read_csv(data_path)

    # clean the data
    filtered_data_cleaned = data_for_content_filtering(filtered_data)

    # transform the data
    transformed_data = transform_data(filtered_data_cleaned)

    # save the transformed data
    save_transformed_data(transformed_data,save_path)


if __name__ == "__main__":
    main(FILTERED_DATA_PATH,SAVE_PATH)