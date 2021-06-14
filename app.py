from flask import Flask
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from pywebio.input import file_upload
from pywebio.output import put_image,put_tabs
from pywebio.platform.flask import webio_view
from pywebio import start_server
import argparse


from joblib import load
labels = load('labels.joblib')

model_name = 'xray_model'

def binary_balanced_accuracy(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    y_true = y_true.ravel()
    y_pred = np.round(y_pred.ravel())
    num_classes = len(np.unique(y_true))

    cm = confusion_matrix(y_true, y_pred).T
    balanced_accuracy = 0
    for i in range(num_classes):
        num = cm[i,i]
        den = np.sum(cm[:,i])
        if num == 0 :
            acc = 0
        else:
            acc = num / den
        balanced_accuracy += acc

    return (balanced_accuracy / num_classes)

model = tf.keras.models.load_model(model_name , custom_objects = {'binary_balanced_accuracy' : binary_balanced_accuracy})

app = Flask(__name__)


def predict_class():
    img = file_upload(placeholder = 'Upload image of the xray')
    x = tf.io.decode_image(img['content'])
    try:
        x = tf.image.rgb_to_grayscale(x)
    except :
        pass
    x = tf.image.resize(x,[150,150])
    x = x / 255.0
    x = tf.expand_dims(x, axis=0)
    cls = model.predict(x)
    cls = np.round(cls.ravel())
    text = 'Predicted Class : ' + str(labels[int(cls[0])])
    put_tabs([
    {'title': 'Result', 'content': text},
    {'title': 'Uploaded Image', 'content': [
        put_image(img['content'],width='300px',height='300px',title='X-ray_Image')
    ]},
    ])

app.add_url_rule('/tool', 'webio_view', webio_view(predict_class),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict_class, port=args.port)   
    
    
#app.run(host='localhost', port=80)
    