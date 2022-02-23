FROM    pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
# FROM    legosz/cartoonify:v1

WORKDIR /app
COPY    . .

# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE  5000
# CMD ["python", "main.py"]