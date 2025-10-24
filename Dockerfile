# 1. Start from an official Python 3.11 base image
FROM python:3.11-slim

# 2. Install system libraries needed by OpenCV
#    We use 'libgl1' instead of the older 'libgl1-mesa-glx'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container to /app
WORKDIR /app

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all your project files
COPY . .

# 7. Tell Docker what command to run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]