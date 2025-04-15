import os
import json
import subprocess
from zipfile import ZipFile
import shutil

class BookSumDataLoader:
    def __init__(self, data_dir=os.path.join('data','booksum'), zip_file='all_chapterized_books.zip', 
                 gs_url='gs://sfr-books-dataset-chapters-research/all_chapterized_books.zip'):
        self.data_dir = data_dir
        self.zip_file = zip_file
        self.gs_url = gs_url

        # Ensure the data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _data_exists(self):
        """Check if the data directory contains the unzipped files."""
        return any(os.scandir(self.data_dir))

    def _rename_folders(self):
        """Rename folders incrementally."""
        i = 0
        for root, dirs, files in os.walk(self.data_dir, topdown=False):
            # Check if the current directory is a folder to be renamed
            if dirs == []:  # Only rename folders without subdirectories
                parent_dir = os.path.dirname(root)
                new_folder_name = os.path.join(parent_dir, f"{i}")

                # Ensure the new folder name does not conflict
                while os.path.exists(new_folder_name):
                    i += 1
                    new_folder_name = os.path.join(parent_dir, f"{i}")

                # Rename the folder
                os.rename(root, new_folder_name)
                i += 1

    def _clean_data(self):
        """Remove all files except 'content.txt'."""
        for root, dirs, files in os.walk(self.data_dir):
            # Renaming 'book_clean.txt' to 'content.txt' if it exists
            if 'book_clean.txt' in files:
                old_file_path = os.path.join(root, 'book_clean.txt')
                new_file_path = os.path.join(root, 'content.txt')
                os.rename(old_file_path, new_file_path)

            # Remove all files except 'content.txt'
            for file in files:
                if file != 'content.txt':
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
    def _move_folders(self):
        """Move all subfolders to the main data directory and remove the root folder."""
        root_folder = os.path.join(self.data_dir, 'all_chapterized_books')
        if os.path.exists(root_folder):
            for item in os.listdir(root_folder):
                item_path = os.path.join(root_folder, item)
                if os.path.isdir(item_path):
                    # Move folder to the main data directory
                    shutil.move(item_path, self.data_dir)

            # Remove the now-empty root folder
            os.rmdir(root_folder)
    
    def download_data(self):
        """Download and unzip the data if not already present."""
        if not self._data_exists():
            print("Data not found. Downloading...")
            # Download the zip file
            subprocess.run(['gsutil', 'cp', self.gs_url, self.zip_file], check=True)
            
            # Unzip the file
            with ZipFile(self.zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            # Remove the zip file
            os.remove(self.zip_file)

            print("Data downloaded, extracted, and zip file removed.")
        
        # Rename folders incrementally
        print("Renaming folders...")
        self._rename_folders()

        # Clean the data to keep only 'book_clean.txt'
        print("Cleaning data...")
        self._clean_data()
        print("Data cleaned.")

        print("Moving folders...")
        self._move_folders()
        print("Data processed.")

    def load_data(self):
        """Main method to ensure data is available and then load it."""
        self.download_data()
        # Add any additional data loading logic here
        # For example, reading 'book_clean.txt' files or processing text, etc.
    def get_number_of_books(self):
        """Return the number of books (subfolders) in the data directory."""
        return len([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
    def get_book_content(self, id):
        """Read the 'content.txt' file for a specified book and return its content."""
        content_file_path = os.path.join(self.data_dir,str(id), 'content.txt')
        if os.path.exists(content_file_path):
            with open(content_file_path, 'r', encoding='utf-8') as file:
                return file.read()
        return None

if __name__=='__main__':
    data_loader = BookSumDataLoader()
    data_loader.load_data()
    print(data_loader.get_number_of_books())
    print(data_loader.get_book_content(32)) # returns the content, id from 0 to 267