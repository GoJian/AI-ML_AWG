#################
# Frank Soboczenski, PhD
# GeneLab AWG AI/ML Working Group
# RR9 data sorting script
################
#
# Note: place it in the folder with the images and the meta.csv file. It will look into the file and pick the class column
# then shuffles the images into separate folders under "train"  according to their class.
#

import os
import pandas as pd
from shutil import move


# Function to sort pictures based on a column in the CSV file
def sort_pictures(csv_filename, picture_folder, column_name):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_filename)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the picture filename and class from the specified column
        picture_filename = row['filename']  # Replace 'Filename' with the actual column name
        picture_class = row[column_name]

        # Create destination folder based on the class
        destination_folder = os.path.join(picture_folder, picture_class)

        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Construct the source and destination paths
        source_path = os.path.join(picture_folder, picture_filename)
        destination_path = os.path.join(destination_folder, picture_filename)

        # Check if source and destination paths are different before moving
        if source_path != destination_path:
            # Move the picture to the appropriate folder
            move(source_path, destination_path)
            print(f"Moved {picture_filename} to {picture_class} folder")
        else:
            print(f"Skipping {picture_filename}: Source and destination paths are the same.")


if __name__ == "__main__":
    # Specify the CSV filename, picture folder, and the column to sort by
    csv_filename = "meta.csv"
    picture_folder = "train"
    column_to_sort_by = "particle_type"  # Replace with the actual column name

    # Call the function to sort pictures
    sort_pictures(csv_filename, picture_folder, column_to_sort_by)
