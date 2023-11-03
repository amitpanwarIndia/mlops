FROM python:3.10.12

WORKDIR /home/caladmin/iitj-mlops/projects/mlops

COPY . /home/caladmin/iitj-mlops/projects/mlops

RUN pip install -r requirements.txt

VOLUME /home/caladmin/iitj-mlops/projects/mlops/models

CMD python ex.py

