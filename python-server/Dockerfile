FROM python:3

WORKDIR /usr/src/python-server

RUN pip install --upgrade pip

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD [ "python", "main.py" ]
