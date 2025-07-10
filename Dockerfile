FROM python:3.10-slim

WORKDIR /app

COPY . .

# RUN apt-get update && apt-get install -y \
#     git \
#     libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN pip install --upgrade pip

#  Install torch and torchvision (CPU)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Rest of requirements
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
