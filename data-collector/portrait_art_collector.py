# portrait_art_collector.py
"""Code that collects portrait images from the MET Open Access API, Wikiart.org, and the Rijksmuseum OAI-PMH endpoint."""
import requests
import os
import time
from oaipmh.client import Client
from oaipmh.metadata import MetadataRegistry, oai_dc_reader
from urllib.parse import urlparse
from bs4 import BeautifulSoup  # For web scraping

def create_directory(dir_name):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created.")

def download_image(image_url, filename):
    """Downloads an image from a URL and saves it to a file."""
    if os.path.exists(filename):
        print(f"Skipping download, file already exists: {filename}")
        return True  # Indicate success as the file is already present

    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()

        if not os.path.abspath(os.path.realpath(filename)).startswith(os.getcwd()):
            raise RuntimeError('Filepath falls outside the base directory')
        with open(filename, 'wb') as outfile:
            for chunk in response.iter_content(chunk_size=8192):
                outfile.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
        return False
# import os


def fetch_met_portraits(output_dir="/efs/dataset/met_portraits", max_images=50):
    """Fetches portrait paintings from the Metropolitan Museum of Art API."""
    print("Fetching portraits from The Met...")
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
    search_url = f"{base_url}/search?q=portrait&departmentIds=11"  # Department ID 11 is for European Paintings
    objects_url = f"{base_url}/objects/"  # To fetch object details

    create_directory(output_dir)
    image_count = 0

    try:
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        object_ids = search_response.json().get('objectIDs', [])

        for object_id in object_ids[:max_images]:  # Limit to max_images for demonstration
            if image_count >= max_images:
                break

            object_detail_url = objects_url + str(object_id)
            detail_response = requests.get(object_detail_url)
            detail_response.raise_for_status()
            object_data = detail_response.json()

            if object_data.get('primaryImage') and object_data.get('isPublicDomain'):
                image_url = object_data['primaryImage']
                title = object_data.get('title', 'Untitled').replace("/", "-")[:100]
                artist = object_data.get('artistDisplayName', 'Unknown Artist').replace("/", "-")[:100]
                filename = os.path.join(output_dir, f"met_{object_id}_{artist}_{title}.jpg")

                if not os.path.exists(filename):
                    if download_image(image_url, filename):
                        image_count += 1
                        print(f"Downloaded: {filename}")
                    else:
                        print(f"Warning: Download failed for: {filename} from {image_url}")
                else:
                    print(f"Skipping download, file already exists: {filename}")
                    image_count += 1  # still count it
                time.sleep(0.5)  # Be polite to the API

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from The Met API: {e}")

    print(f"Downloaded {image_count} portraits from The Met to '{output_dir}'.")


def fetch_rijksmuseum_oai_portraits(output_dir="/efs/dataset/rijksmuseum_oai_portraits", max_images=50):
    """Fetches portrait paintings from Rijksmuseum using OAI-PMH."""
    print("Fetching portraits from Rijksmuseum using OAI-PMH...")
    base_url = "https://data.rijksmuseum.nl/oai"

    create_directory(output_dir)
    image_count = 0
    record_count = 0
    max_record_harvest = max_images * 5  # Harvest more records initially, then filter

    try:
        # Setup OAI-PMH client
        metadata_registry = MetadataRegistry()
        metadata_registry.registerReader(oai_dc_reader())  # Register Dublin Core reader
        client = Client(base_url, metadata_registry)

        # Harvest records in 'oai_dc' format
        records = client.listRecords(metadataPrefix='oai_dc')

        for record_tuple in records:
            if image_count >= max_images or record_count >= max_record_harvest:
                break

            header, metadata, errors = record_tuple
            record_count += 1

            if errors:
                print(f"Error processing record {header.identifier()}: {errors}")
                continue

            # Extract metadata
            identifier = header.identifier()
            title_list = metadata.get('title')
            creator_list = metadata.get('creator')
            description_list = metadata.get('description')
            subject_list = metadata.get('subject')
            type_list = metadata.get('type')
            rights_list = metadata.get('rights')

            title = title_list[0] if title_list else "Untitled"
            creator = creator_list[0] if creator_list else "Unknown Artist"
            description = description_list[0] if description_list else "No description"
            subjects = subject_list if subject_list else []
            types = type_list if type_list else []
            rights = rights_list[0] if rights_list else "Unknown Rights"

            # Filter for portraits
            is_portrait = False
            for subject in subjects:
                if "portrait" in subject.lower():
                    is_portrait = True
                    break
            if not is_portrait:
                for desc in description.split('.'):
                    if "portrait" in desc.lower():
                        is_portrait = True
                        break
            if not is_portrait:
                continue

            # Attempt a speculative image URL
            webpage_url_base = "https://www.rijksmuseum.nl/en/collection/"
            webpage_url = webpage_url_base + identifier.split(':')[-1]
            parsed_url = urlparse(webpage_url)
            path_components = parsed_url.path.split('/')
            object_number_from_url = path_components[-1] if path_components[-1] else None

            if object_number_from_url:
                speculative_image_url = f"https://www.rijksmuseum.nl/thumb/rkd/xxl/{object_number_from_url}.jpg"
                filename = os.path.join(
                    output_dir,
                    f"rijksmuseum_oai_{identifier.split(':')[-1]}_{creator.replace('/', '-')[:100]}_{title.replace('/', '-')[:100]}.jpg"
                )

                if not os.path.exists(filename):
                    if download_image(speculative_image_url, filename):
                        image_count += 1
                        print(f"Downloaded: {filename} from (speculative URL: {speculative_image_url}), Title: {title}, Creator: {creator}")
                    else:
                        print(f"Warning: Download failed for: {filename} from {speculative_image_url}")
                else:
                    print(f"Skipping download, file already exists: {filename}")
                    image_count += 1
                time.sleep(0.5)

    except Exception as e:
        print(f"Error fetching data from Rijksmuseum OAI-PMH: {e}")

    print(f"Downloaded {image_count} portrait images from Rijksmuseum OAI-PMH to '{output_dir}'.")
    print(f"Harvested and processed {record_count} records in total (before portrait filter).")


def fetch_wikiart_portraits(output_dir="/efs/dataset/wikiart_portraits", max_images=50):
    """Scrapes portrait paintings from Wikiart.org (using genre category) with pagination."""
    print("Fetching portraits from Wikiart.org (genre 'portrait')...")
    base_url = "https://www.wikiart.org"
    genre_url = base_url + "/en/paintings-by-genre/portrait"

    create_directory(output_dir)
    image_count = 0
    page_num = 1

    while image_count < max_images:
        current_page_url = f"{genre_url}/{page_num}"
        print(f"Scraping page: {current_page_url}")

        try:
            response = requests.get(current_page_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Example selectors - may need adjustment
            artwork_items = soup.find_all('li', class_='artwork-item')
            if not artwork_items:
                print(f"No artwork items found on page {page_num}. Stopping scraping.")
                break

            for item in artwork_items:
                if image_count >= max_images:
                    break

                image_link_element = item.find('a', class_='artwork-link')
                if image_link_element and image_link_element.has_attr('href'):
                    artwork_page_url = base_url + image_link_element['href']

                    try:
                        artwork_response = requests.get(artwork_page_url, timeout=10)
                        artwork_response.raise_for_status()
                        artwork_soup = BeautifulSoup(artwork_response.content, 'html.parser')

                        # Example selector - may need adjustment
                        image_element = artwork_soup.find('img', class_='ms-zoom-image-main')
                        if image_element and image_element.has_attr('src'):
                            image_url = image_element['src']
                            if not image_url.startswith('http'):
                                image_url = "https:" + image_url

                            title_element = artwork_soup.find('h1', class_='artwork-title')
                            artist_element = artwork_soup.find('a', class_='artist-name')

                            title = title_element.text.strip() if title_element else "Untitled"
                            artist = artist_element.text.strip() if artist_element else "Unknown Artist"

                            filename = os.path.join(
                                output_dir,
                                f"wikiart_{page_num}_{image_count + 1}_"
                                f"{artist.replace('/', '-')[:100]}_"
                                f"{title.replace('/', '-')[:100]}.jpg"
                            )

                            if not os.path.exists(filename):
                                if download_image(image_url, filename):
                                    image_count += 1
                                    print(f"Downloaded: {filename}, Page: {page_num}, Count: {image_count}, "
                                          f"Title: {title}, Artist: {artist}")
                                else:
                                    print(f"Warning: Download failed for: {filename} from {image_url}")
                            else:
                                print(f"Skipping download, file already exists: {filename}")
                                image_count += 1
                            time.sleep(3)  # Adjust delay

                        else:
                            print(f"Warning: Could not find image URL on artwork page: {artwork_page_url}")

                    except requests.exceptions.RequestException as artwork_err:
                        print(f"Warning: Error fetching artwork page {artwork_page_url}: {artwork_err}")

        except requests.exceptions.RequestException as page_err:
            print(f"Error fetching genre page {current_page_url}: {page_err}")
            break

        if image_count >= max_images:
            print(f"Reached max_images limit ({max_images}). Stopping pagination.")
            break

        page_num += 1
        time.sleep(2)

    print(f"Downloaded {image_count} portrait images from Wikiart to '{output_dir}'.")


if __name__ == "__main__":
    # Point everything to an EFS-mounted directory at /efs/dataset
    output_directory = "/efs/dataset/portrait_dataset_oai_wikiart_paged"
    create_directory(output_directory)

    # Example calls - adjust as needed
    fetch_met_portraits(
        output_dir=os.path.join(output_directory, "met_images"),
        max_images=200
    )
    fetch_wikiart_portraits(
        output_dir=os.path.join(output_directory, "wikiart_images"),
        max_images=200
    )
    fetch_rijksmuseum_oai_portraits(
        output_dir=os.path.join(output_directory, "rijksmuseum_oai_images"),
        max_images=20
    )

    print("\nPortrait image collection using OAI-PMH and Wikiart scraping complete.")
    print(f"Images are saved in the '{output_directory}' directory.")
