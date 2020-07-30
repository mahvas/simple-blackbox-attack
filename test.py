#!/usr/bin/env python

import argparse
import requests
import numpy
import json

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, help='file to test', default='bulbul.jpg')

args = parser.parse_args()

data = open(args.infile,'rb').read()

res = requests.post(url='http://817c1c6b-37b5-43a4-a947-9e252b665ead.westeurope.azurecontainer.io/score', 
                    data=data, headers={'Content-Type':'application/octet-stream'})
class_dict=json.loads(open('imagenet_class_index.json', 'r').read())


out = json.loads(res.text)

predicted_index=numpy.argmax(out['result'])

print(f'predicted class{class_dict[str(predicted_index)]} \
        with index {predicted_index} \
        with probability {out["result"][predicted_index]}')