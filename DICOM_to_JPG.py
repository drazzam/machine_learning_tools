# Import libraries
import os
from urllib.request import urlretrieve
import pydicom
from PIL import Image
import zipfile
------------------------------------------------
# Set the URL of the DICOM image file
url = 'ENTER_THE_DICOM_FILE_URL'

# Set the destination filename
filename = 'ENTER_THE_DICOM_FILE_NAME'

# Download the DICOM image from the URL
urlretrieve(url, filename)

# Read the DICOM image file
ds = pydicom.dcmread(filename)

# Convert the image to RGB mode
img = img.convert('RGB')

# Save the image as a JPEG file
img.save('image.jpg')

# Compress the JPG image to reduce its file size
img.save('image.jpg', optimize=True, quality=85)

# Create a ZIP file containing the JPG image
with zipfile.ZipFile('images.zip', 'w') as zip:
 zip.write('image.jpg', compress_type=zipfile.ZIP_DEFLATED)
