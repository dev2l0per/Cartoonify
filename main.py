from queue import Empty, Queue
import threading
from flask import (
    Flask, request, Response, send_file
)

import torch
import time
from torchvision import transforms
from src.models.generator import Generator
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
from torchvision.utils import make_grid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_DIM         = 512
lr              = 2e-4
MAPS_GEN        = 64
MAPS_DISC       = 64
IMG_CHANNELS    = 3
L1_LAMBDA       = 100

Transforms = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))
])

model = Generator(img_channels=3, features=64)
model.load_state_dict(torch.load('./weights/Cartoonify_Generator.pt', map_location=torch.device(DEVICE)))
model.eval()

app = Flask(__name__)

requestsQueue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1

def handle_requests_by_batch():
    while True:
        requestsBatch = []
        while not (len(requestsBatch) >= BATCH_SIZE):
            try:
                requestsBatch.append(requestsQueue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for request in requestsBatch:
                request['output'] = generate(request['input'][0])

def generate(file):
    try:
        binary = file.read()
        img = Image.open(BytesIO(binary))
        transformImg = Transforms(img)

        out = model(transformImg.unsqueeze(0).to(DEVICE))
        imgGridFake = make_grid(out[:8], normalize=True)
        imgGridFake = imgGridFake.cpu().detach().numpy()
        out = np.transpose(imgGridFake, (1, 2, 0))
        toPILImage = transforms.ToPILImage()
        out = toPILImage(np.uint8(out*255))
        bufferOut = BytesIO()
        out.save(bufferOut, format=f'{file.content_type.split("/")[-1]}')
        bufferOut.seek(0)
        
        return bufferOut
    except Exception as e:
        return "error"

@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    try:
        file = request.files['file']
    except:
        return Response("Empty Field", status=400)
    
    req = {
        'input': [file],
    }

    requestsQueue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)
    
    io = req['output']
    if io == "error":
        return Response('Server Error', status=500)

    return send_file(io, mimetype=file.content_type)

threading.Thread(target=handle_requests_by_batch).start()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5000")