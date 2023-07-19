import csv
import os
import urllib.request

# Function to download an image from a URL and save it with a specified filename
def download_image(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# Name of the TSV file
tsv_file = 'Q:\Documents\Berkeley\Capstone\trulia_image_list_filtered.txt'

# Directory to save the downloaded images
output_directory = 'Q:\Documents\Berkeley\Capstone\image_data'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Read the TSV file
with open(tsv_file, 'r') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        home_id = row['Home_ID']
        image_urls = [(column_name, row[column_name]) for column_name in row.keys() if column_name.startswith('Image')]
        
        # Create a directory for each home_id
        home_directory = os.path.join(output_directory, home_id)
        os.makedirs(home_directory, exist_ok=True)
        
        # Download and save each image in the home_directory
        for image_name, image_url in image_urls:
            filename = os.path.join(home_directory, f"{image_name}.jpg")
            download_image(image_url, filename)