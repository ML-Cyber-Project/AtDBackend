FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV AWS_ACCESS_KEY_ID=''
ENV AWS_SECRET_ACCESS_KEY=''
ENV AWS_ENDPOINT_URL=''

EXPOSE 8000

CMD ["/bin/sh", "-c", "/app/entrypoint.sh"]
