# Import the necessary modules
from PIL import Image
import os
----------------------------------------------------------
# Set the maximum quality for the compressed image
max_quality = 90

# Set the input and output image paths
input_image_path = 'Enter The Input In Form of JPG File'
output_image_path = 'Enter The Output In Form of JPG File'

# Open the input image and compress it with the given quality
with Image.open(input_image_path) as image:
    image.save(output_image_path, quality=max_quality)

# Check the size of the output image
output_image_size = os.stat(output_image_path).st_size
print(f'Output image size: {output_image_size / 1024} kb')
