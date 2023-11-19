FROM python:3.10.12

COPY . /digits

WORKDIR /digits/

RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Define environment variables
ENV FLASK_APP=/digits/flask/app.py
ENV FLASK_RUN_HOST=0.0.0.0

#VOLUME /digits/models

CMD ["flask", "run"]
