FROM python:3.11-slim


WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

VOLUME /usr/src/app/dati
VOLUME /usr/src/app/risultati