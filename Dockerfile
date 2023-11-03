FROM python:3.10.12

COPY . /digits

WORKDIR /digits/

RUN pip install -r requirements.txt

VOLUME /digits/models

CMD ["python", "ex.py"]
