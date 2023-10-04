FROM python:3.9
RUN pip install --upgrade pip setuptools
RUN pip install flwr
WORKDIR /app
COPY . /app
CMD ["python3", "server.py"]
