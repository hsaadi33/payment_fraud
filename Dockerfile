FROM python:3:10

COPY requirements.txt .

WORKDIR . /home/payment_fraud

RUN pip install -r requirement.txt

WORKDIR /home/payment_fraud

CMD ["fastapi", "run", "app/main.py", "--port", "80"]