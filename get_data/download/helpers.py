import logging
import os
import urllib.request
import zipfile


def mkdir_if_not_exists(loc, file=False):
    """Makes a directory if it doesn't already exist. If loc is specified as a file, ensure the file=True option is set.
    Args:
        loc (str): The file/folder for which the folder location needs to be created.
        file (bool): Set true if supplying a file (then will get the dirstring and make the dir).
    Returns:
        True if exists, False if did not exist before.
    """
    existed = True
    loc_ = os.path.dirname(loc) if file else loc
    if not os.path.exists(loc):
        os.makedirs(loc_, exist_ok=True)
        existed = False
    return existed


def download_url(url: str, loc: str):
    """Downloads a url file to a specified location.

    Args:
        url (str): The url location.
        loc (str): The full file location string.
    """
    if not os.path.exists(loc):
        urllib.request.urlretrieve(url, loc)


def unzip(file: str, loc: str):
    """Unzip a folder into a specified location."""
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(loc)


def download_zip(folder: str, name: str, url: str, unzip: bool = True):
    """Downloads a zipfile and extracts if specified.

    Args:
        folder (str): The folder in which to download the zipfile.
        name (str): The name of the zipfile.
        url (str): The url location of the zipfile.
        unzip (bool): Set True to automatically unzip after download.
    """
    assert os.path.isdir(
        folder
    ), "Please make a folder at {} to store the data.".format(folder)

    # Ignore if folder already contains files
    location_for_zip = "{}/{}.zip".format(folder, name)
    if len(os.listdir(folder)) > 0:
        logging.info(
            "Files exist in {}. Delete to restart the download process.".format(folder)
        )
        return None

    download_url(url, location_for_zip)
    if unzip:
        unzip(location_for_zip, folder)
