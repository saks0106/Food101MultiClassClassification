from keras.models import load_model
from PIL import Image
import streamlit as st
from util import load_and_prep_image
import time
import tensorflow as tf


class Model:

    def __init__(self):
        self.model = load_model('./model/efficientnet_v2.h5')

    def image_display(self, file):
        if file is not None:
            image = Image.open(file).convert('RGB')
            st.markdown(f'<h2>Image Uploaded! Scroll below the Image to get Classification Results!</h2>', unsafe_allow_html=True)

            st.image(image,width=900)
            self.streamlit_clf(image)

    def image_display_url(self, image):
            st.markdown(f'<h2>Image Uploaded! Scroll below the Image to get Classification Results!</h2>', unsafe_allow_html=True)
            st.image(image, width=900)
            self.streamlit_clf(image)

    def streamlit_clf(self, image):
        # load class names
        with open('labels.txt', 'r') as f:
            class_names = [a.rstrip() for a in f.readlines()]
            f.close()

            img = load_and_prep_image(image, scale=False)
            pred_prob = self.model.predict(
                tf.expand_dims(img, axis=0))  # make prediction on image with shape [None, 224, 224, 3]
            pred_class = class_names[pred_prob.argmax()]  # find the predicted class label with highest probability
            st.write('#')
            pred_prob = pred_prob.max() * 100
            st.write(f"# Model Thinks the Dish is {pred_class} 🍲 with Confidence level of {pred_prob:.2f}% 😋😄 ")
            st.markdown('#')
            time.sleep(0.5)
            st.balloons()
            time.sleep(0.9)
            st.snow()
