FROM python:3.9
RUN pip install --upgrade pip setuptools 
RUN pip install numpy flwr torch torchvision
WORKDIR /app
COPY . /app
CMD ["python3", "client.py"]
