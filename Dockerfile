# pull official base image
FROM python:3.8.6

RUN apt-get install libsndfile1

RUN pip install --no-cache datasets transformers soundfile numpy torch torchaudio
RUN pip install -e .

COPY . .
