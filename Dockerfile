FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# --- Dépendances système ---
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    libgdal-dev \
    libqt5gui5 \
    libqt5core5a \
    libqt5printsupport5 \
    libqt5svg5 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# --- Dépôt QGIS ---
RUN wget -O - https://qgis.org/downloads/qgis-archive-keyring.gpg | gpg --dearmor | tee /etc/apt/keyrings/qgis-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu jammy main" > /etc/apt/sources.list.d/qgis.list

# --- Installer QGIS ---
RUN apt-get update || (sleep 10 && apt-get update) \
    && apt-get install -y \
    qgis \
    qgis-server \
    qgis-plugin-grass \
    python3-qgis \
    qgis-providers \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Variables d'environnement QGIS ---
ENV QGIS_PREFIX_PATH="/usr"
ENV PYTHONPATH=/usr/share/qgis/python
ENV QT_QPA_PLATFORM=offscreen
ENV QT_DEBUG_PLUGINS=0

# Copier ton code
COPY . /app

# Installer requirements Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Lancer Django
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
