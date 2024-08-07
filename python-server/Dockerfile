FROM python:3

WORKDIR /usr/src/python-server

RUN apt -y update && apt -y upgrade
RUN apt -y install libopencv-dev
RUN pip install --upgrade pip


COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 9000/udp
CMD [ "python", "main.py" ]
