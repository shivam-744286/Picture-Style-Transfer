import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Function to load and preprocess the image
def load_image(path_to_img, max_dim=512):
    img = Image.open(path_to_img)
    img = img.convert('RGB')
    img = img.resize((max_dim, max_dim))
    img = np.array(img)
    img = img.astype(np.float32)[np.newaxis, ...] / 255.
    return tf.image.resize(img, (max_dim, max_dim))

# Load content and style images

a=input("enter content image name with .jpg extention\n")
content_image = load_image(a)
b=input("enter style image name with .jpg extention\n")
style_image = load_image(b)

# Load the pre-trained style transfer model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Apply style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Display the result
def show_image(image, title=''):
    plt.imshow(tf.squeeze(image))
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image(content_image, 'Content Image')
show_image(style_image, 'Style Image')
show_image(stylized_image, 'Stylized Image')
