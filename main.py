import streamlit as st
from keras.models import load_model
from PIL import Image
import time
import tensorflow as tf
from util import load_and_prep_image

st.set_page_config(page_title='Food101',
                   page_icon='🍚',
                   layout='centered'
                   # initial_sidebar_state='collapsed',
                   # menu_items={'Get Help': 'https://www.google.com'}
                    )


# set title
st.title('Welcome to Food101 MultiClass Classification ! :100:🥘🥙😍😋',anchor='xyz')

st.write("#")
# set header
st.subheader('Please upload an image of a Dish🥣 you want to predict🤔')
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/efficientnet_v2.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a.rstrip() for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)
    st.write('#')
    img = load_and_prep_image(image,scale=False)
    pred_prob = model.predict(tf.expand_dims(img, axis=0))  # make prediction on image with shape [None, 224, 224, 3]
    pred_class = class_names[pred_prob.argmax()]  # find the predicted class label with highest probability
    st.write('#')
    pred_prob = pred_prob.max() * 100
    st.write(f"# Model Thinks the Dish is {pred_class} 🍲 with Confidence level of {pred_prob:.2f}% 😋😄 ")
    st.markdown('#')
    time.sleep(0.5)
    st.balloons()
    time.sleep(0.9)
    st.snow()


