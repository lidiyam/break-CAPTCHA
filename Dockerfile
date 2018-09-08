# Dockerfile with tensorflow gpu support on python3, opencv3.3
FROM fbcotter/docker-tensorflow-opencv:latest

WORKDIR /app
COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY data/ .
COPY extract_single_letters_from_captchas.py .
RUN python3 extract_single_letters_from_captchas.py

COPY . .
