# ref: https://docs.ovh.com/ie/en/publiccloud/ai/apps/tuto-gradio-sketch-recognition/

import PIL
import gradio as gr


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model


img_size=32
labels=["ZERO","ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE"]

model=load_model('BanglaModel.h5')

def preprocessing(img):
    # img=img.astype("uint8")
    # img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img = img/255
    return img

def predict(img):
    if img is not None:
        img=cv2.resize(img,(img_size,img_size))
        img=preprocessing(img)
        img=img.reshape(1,img_size,img_size,1)

        preds=model.predict(img)[0]

        return {label: float(pred) for label, pred in zip(labels, preds)}
    else:
        return None


if __name__=="__main__":
    label=gr.outputs.Label(num_top_classes=3)
    input_img = gr.inputs.Image(
                  image_mode='L', 
                  source='canvas', 
                  shape=(32, 32), 
                  tool= 'select')
    app=gr.Interface(
        fn=predict,
        inputs=input_img,
        outputs=label,
        title="Bangla HandWriting Digit Classification",
        description="Bangla HandWriting Digit Classification",
        live=True,
    )
    app.launch(share=True)