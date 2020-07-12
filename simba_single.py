import argparse
import numpy as np
import requests
import torch
import torchvision
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import utils
from PIL import Image

def normalize(x):
    return utils.apply_normalization(x, 'imagenet')

# def get_probs(model, x, y):
#     output = model(normalize(x.cuda())).cpu()
#     probs = torch.nn.Softmax()(output)[:, y]
#     return torch.diag(probs.data)

def get_probs(x, y):
    torchvision.utils.save_image(x, fp='/tmp/out.jpg')
    with open('/tmp/out.jpg', mode='rb') as out:
        response = requests.post(url='http://817c1c6b-37b5-43a4-a947-9e252b665ead.westeurope.azurecontainer.io/score', data=out.read(), headers={'Content-Type':'application/octet-stream'})
    score = response.json()['result'][y]
    print(f'Score: {score}')
    return score
    

# 20-line implementation of (untargeted) SimBA for single image input
def simba_single(x, y, num_iters=100, epsilon=0.2):
    n_dims = x.view(1, -1).size(1)
    last_prob = get_probs(x, y)
    print('run simba_single')
    for i in range(num_iters):
        perm = torch.randperm(n_dims)        
        print(f'starting iteration {i}')
        diff = torch.zeros(n_dims)
        for j in range(50000):
            diff[perm[i - j]] = epsilon
        left_prob = get_probs((x - diff.view(x.size()).clamp(0, 1)), y)
        if left_prob < last_prob:
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob = left_prob
        else:
            right_prob = get_probs((x + diff.view(x.size()).clamp(0, 1)), y)
            if right_prob < last_prob:
                x = (x + diff.view(x.size())).clamp(0, 1)
                last_prob = right_prob
    return x

def main():
    print('run main of simba_single')
    parser = argparse.ArgumentParser()
    with Image.open('/Users/vashishthamahesh/Documents/imagenet/val/bulbul/bulbul.jpg') as image:
        simba_single(ToTensor()(image), 16)

if __name__ == "__main__":
    main()