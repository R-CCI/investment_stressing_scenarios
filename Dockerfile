FROM python:3.10

EXPOSE 8080
WORKDIR /app

COPY . ./

RUN apt-get update && apt-get install -y tesseract-ocr
RUN pip install --upgrade pip setuptools wheel
RUN pip install "rich<14"
RUN pip install --upgrade streamlit
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "AnálisisInversiones.py", "--server.port=8080", "--server.address=0.0.0.0"]
