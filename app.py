# import streamlit as st
# from cnn_model import Model
# from PIL import Image
# import time
# import tensorflow as tf
# from util import load_and_prep_image
# import requests
# import os
# from os import listdir
# from pathlib import Path
# from PIL import Image
# import requests
# from io import BytesIO
#
# import requests
# from PIL import Image
# from io import BytesIO
#
# st.set_page_config(page_title='Food101',
#                    page_icon='🍚',
#                    layout='wide'
#                     )
#
#
# # set title
# st.title('Welcome to Food101 MultiClass Classification ! :100:🥘🥙😋',anchor='xyz')
# st.image('./Explaination_ScreenShots/Cover.png')
#
# st.write("#")
# st.subheader('Please upload an image of a Dish🥣 you want to predict🤔 or paste an Image Url 🔗 or Choose from Existing Images 🥩 ')
#
# st.info('This Web Application has been Trained 50k+ Image Dataset and it can classify 101 Food Images :100:')
# st.warning(' Kindly Upload the Correct Link and Correct Image for Best Results! 🤔',icon='✅')
# st.warning('Please wait Patiently while the model :robot_face: is Running :running:')
#
# img_select = st.radio('Select Any from Below Options:',
#                   options =('Upload an Image', 'Choose from Existing Images','Paste a Link'))
#
# model = Model()
#
# if img_select == 'Upload an Image':
#     file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
#     model.image_display(file)
#
# elif img_select == 'Choose from Existing Images':
#     folder_dir = "./images/"
#     img_select = st.radio('Select Any of the Food Dishes Below:',
#                           options=('','burger.jpg', 'butter_chicken.jpg', 'pizza.jpg'),index=0)
#     if img_select == '':
#         pass
#     elif img_select == 'burger.jpg':
#         file = os.path.join(folder_dir, img_select)
#         model.image_display(file)
#
#     elif img_select == 'butter_chicken.jpg':
#         file = os.path.join(folder_dir, img_select)
#         model.image_display(file)
#     else:
#         file = os.path.join(folder_dir, img_select)
#         model.image_display(file)
#
# else:
#         def open_image_from_url(url):
#             try:
#                 response = requests.get(url)
#                 # Check if the request was successful (status code 200)
#                 response.raise_for_status()
#
#                 # Convert the image data to a PIL Image
#                 img = Image.open(BytesIO(response.content))
#                 return img
#
#             except Exception as e:
#                 print(f"Failed to open image from URL: {e}")
#                 return None
#
#
#         # Example usage
#         image_url = st.text_area('Enter Image URL: ', max_chars=1000, placeholder='sample txt')
#         st.warning(' Kindly Upload the Correct and Valid Link for Best Results! 🤔',icon='✅')
#         if st.button('Upload Valid Image Address/Url'):
#             image = open_image_from_url(image_url)
#             model.image_display_url(image)
#
#
#
#


import streamlit as st
from cnn_model import Model
import os
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title='Food101',
                   page_icon='🍚',
                   layout='wide'
                    )


# set title
st.title('Welcome to Food101 MultiClass Classification ! :100:🥘🥙😋',anchor='xyz')
st.image('./Explaination_ScreenShots/Cover.png')

st.write("#")
st.subheader('Please upload an image of a Dish🥣 you want to predict🤔 or paste an Image Url 🔗 or Choose from Existing Images 🥩 ')

st.info('This Web Application has been Trained 50k+ Image Dataset and it can classify 101 Food Images :100:')
st.warning(' Kindly Upload the Correct Link and Correct Image for Best Results! 🤔',icon='✅')
st.warning('Please wait Patiently while the model :robot_face: while the model will be Running :running:')

img_select = st.radio('Select Any from Below Options:',
                  options =('Upload an Image', 'Choose from Existing Images','Paste a Link'))

model = Model()

if img_select == 'Upload an Image':
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
    model.image_display(file)
#
# elif img_select == 'Choose from Existing Images':
#     folder_dir = "./images/"
#     img_select = st.radio('Select Any of the Food Dishes Below:',
#                           options=('','burger.jpg', 'butter_chicken.jpg', 'pizza.jpg'),index=0)
#     if img_select == '':
#         pass
#     elif img_select == 'burger.jpg':
#         file = os.path.join(folder_dir, img_select)
#         model.image_display(file)
#
#     elif img_select == 'butter_chicken.jpg':
#         file = os.path.join(folder_dir, img_select)
#         model.image_display(file)
#     else:
#         file = os.path.join(folder_dir, img_select)
#         model.image_display(file)

# else:
#         def open_image_from_url(url):
#             try:
#                 response = requests.get(url)
#                 # Check if the request was successful (status code 200)
#                 response.raise_for_status()
#
#                 # Convert the image data to a PIL Image
#                 img = Image.open(BytesIO(response.content))
#                 return img
#
#             except Exception as e:
#                 print(f"Failed to open image from URL: {e}")
#                 return None
#
#
#         # Example usage
#         image_url = st.text_area('Enter Image URL: ', max_chars=1000, placeholder='sample txt')
#         st.warning(' Kindly Upload the Correct and Valid Link for Best Results! 🤔',icon='✅')
#         if st.button('Upload Valid Image Address/Url'):
#             image = open_image_from_url(image_url)
#             model.image_display_url(image)




