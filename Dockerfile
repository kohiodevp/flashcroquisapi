# Utiliser l'image officielle QGIS comme base
FROM qgis/qgis:release-3_34

# Installer Python + Uvicorn + dépendances de l’API
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copier ton code API dans /app
WORKDIR /app
COPY . /app

# Installer dépendances Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Démarrage de l’API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
