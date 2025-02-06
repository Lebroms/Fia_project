FROM python:3.11-slim

WORKDIR /usr/src/app

COPY . .

RUN python -m venv ./env

ENV VIRTUAL_ENV /env
ENV PATH /usr/src/app/env/bin:$PATH

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

VOLUME /usr/src/app/dati
VOLUME /usr/src/app/risultati