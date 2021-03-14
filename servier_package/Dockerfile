FROM python:3.6
RUN mkdir /app
RUN pip install flask
RUN pip install numpy scipy
RUN pip install pandas sklearn
COPY app.py /app
COPY serve.py /app
COPY servier /app/servier
COPY setup.py /app
WORKDIR /app
RUN python /app/setup.py develop &&  python /app/app.py
