# -*- coding: utf-8 -*-
import io
import sys
import json
import os
import warnings
import flask

import boto3

import mxnet as mx
from mxnet import nd, image

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model
from mxnet.gluon import nn

from PIL import Image

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

# model_dir = '/opt/ml/model'
model_dir = 'model'

num_gpus = 0
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
print(ctx)

# Load Model
model_name = 'ResNet50_v2'
saved_params = os.path.join(model_dir, 'model-0000.params')
if not os.path.exists(saved_params):
    saved_params = ''
pretrained = True if saved_params == '' else False

if not pretrained:
    classes = [i for i in range(5)]
    net = get_model(model_name, classes=len(classes), pretrained=pretrained)
    net.load_parameters(saved_params)
    
else:
    net = get_model(model_name, pretrained=pretrained)
    classes = net.classes
    
net.collect_params().reset_ctx(ctx)

print(len(net.features))
seq_net = nn.Sequential()
for i in range(len(net.features)):
    seq_net.add(net.features[i])

def get_embedding_advance(input_pic):
    # Load Images
    img = image.imread(input_pic)

    # Transform
    img = transform_eval(img).copyto(ctx[0])
    
    pred = None
    use_layers = [len(seq_net)-1]  # [i for i in range(len(seq_net))]
    for i in range(len(seq_net)):
        img = seq_net[i](img)
        if i in use_layers:
#             print(img.shape)
            pred = img[0]
            break

    return pred.asnumpy().tolist()
    
@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    # print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    
    if flask.request.content_type == 'application/x-image':
        image_as_bytes = io.BytesIO(flask.request.data)
        img = Image.open(image_as_bytes)
        download_file_name = 'tmp.jpg'
        img.save(download_file_name)
        print ("<<<<download_file_name ", download_file_name)
    else:
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
    #     print(data)

        bucket = data['bucket']
        image_uri = data['image_uri']

        download_file_name = image_uri.split('/')[-1]
        print ("<<<<download_file_name ", download_file_name)

        try:
            s3_client.download_file(bucket, image_uri, download_file_name)
            print('Download finished!')
        except:
            #local test
            download_file_name = './1.jpg'

        print ("<<<<download_file_name ", download_file_name)
    
    result = get_embedding_advance(download_file_name)
    
    _payload = json.dumps({'predictions': [result]}, ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')
