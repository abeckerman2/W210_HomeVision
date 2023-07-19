import os
import csv
import requests
import time

# Replace 'YOUR_API_KEY' with your actual Google Maps API key
API_KEY = 'AIzaSyCvjTEUg431ZZSVGL_uVp6tnvWnn9ntfWo'
ZOOM = 20  # The highest zoom level possible for satellite images
OUTPUT_FOLDER = 'satellite_images'
DELAY_SECONDS = 2

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def download_satellite_image(lat, lon, home_id):
    image_path = os.path.join(OUTPUT_FOLDER, f'{home_id}.jpg')
    if os.path.exists(image_path):
        print(f'Satellite image {home_id} already downloaded.')
        return

    base_url = 'https://maps.googleapis.com/maps/api/staticmap?'
    params = {
        'center': f'{lat},{lon}',
        'zoom': ZOOM,
        'size': '640x640',
        'maptype': 'satellite',
        'key': API_KEY,
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open(image_path, 'wb') as f:
            f.write(response.content)
        print(f'Satellite image {home_id} downloaded successfully.')
    else:
        print(f'Failed to download satellite image {home_id}. Status code: {response.status_code}')
    time.sleep(DELAY_SECONDS)

def main():
    with open('data_cleaned_filtered.csv', 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            home_id = row['Home_ID']
            lat, lon = float(row['Latitude']), float(row['Longitude'])
            download_satellite_image(lat, lon, home_id)
#             time.sleep(DELAY_SECONDS)

if __name__ == '__main__':
    main()