import argparse
import numpy as np
import pathlib
import io
import requests
import torch
import torchvision
from torchvision import transforms
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

def get_probs_locally(model, x, y):
    torchvision.utils.save_image(x, fp='/tmp/out.jpg')
    preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image.open('/tmp/out.jpg'))
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)    
    probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy().tolist()    
    score = probabilities[y]
    print(f'bulbul Score: {score}')
    print('top indices and scores: ' + str(sorted([(tup[0], tup[1]) for tup in enumerate(probabilities)], key=lambda x : x[1])[-10:]))
    return score    

def get_probs_with_rest_request(x, y):
    torchvision.utils.save_image(x, fp='/tmp/out.jpg')
    with open('/tmp/out.jpg', mode='rb') as out:
        response = requests.post(url='http://817c1c6b-37b5-43a4-a947-9e252b665ead.westeurope.azurecontainer.io/score', data=out.read(), headers={'Content-Type':'application/octet-stream'})
    score = response.json()['result'][y]
    print(f'bulbul score: {score}')
    print('top indices and scores: ' + str(sorted([(tup[0], tup[1]) for tup in enumerate(response.json()['result'])], key=lambda x : x[1])[-10:]))
    return score

def get_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)    
    pathlib.Path('/tmp/model_dir').mkdir(parents=True, exist_ok=True)
    model_path = '/tmp/model_dir/gopalv-resnet50'
    model.eval()
    torch.save(model.state_dict(), model_path)
    return model

# 20-line implementation of (untargeted) SimBA for single image input
def simba_single(x, y, num_iters=1000, epsilon=0.2):
    model = get_model()
    get_probs = lambda x, y : get_probs_locally(model, x, y)
    n_dims = x.view(1, -1).size(1)
    perm = torch.randperm(n_dims)
    last_prob = get_probs(x, y)
    all_last_probs = [last_prob]
    with open('/tmp/simba_iters.txt', mode='wt') as log:
        for i in range(num_iters):
            print(f'starting iteration {i}')
            diff = torch.zeros(n_dims)
            diff[perm[i]] = epsilon
            left_prob = get_probs((x - diff.view(x.size()).clamp(0, 1)), y)
            if left_prob < last_prob:
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = get_probs((x + diff.view(x.size()).clamp(0, 1)), y)
                if right_prob < last_prob:
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            all_last_probs.append(last_prob)
            log.write(f'after iteration {i}: probability {last_prob}\n')
        print('All probabilities, including before any iterations: ' + str(all_last_probs))
        log.write(f'{all_last_probs}\n')
    return x

def main():
    print('run main of simba_single')
    parser = argparse.ArgumentParser()
    with Image.open('/Users/vashishthamahesh/Documents/imagenet/val/bulbul/bulbul_gopal.jpg') as image:    
    # with Image.open('/Users/vashishthamahesh/Documents/imagenet/val/bulbul/bulbul.jpg') as image:
        simba_single(ToTensor()(image), 16)

if __name__ == "__main__":
    main()