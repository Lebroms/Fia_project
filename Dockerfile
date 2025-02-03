# Usa un'immagine base con Python 3.12
FROM python:3.12-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia i file del progetto (codice, dataset, requirements.txt)
COPY . .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Crea la cartella per i risultati
RUN mkdir -p results

# Comando per eseguire lo script
CMD ["python", "main.py"]