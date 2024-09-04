FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY 2nd_hand_fashion_valuation 2nd_hand_fashion_valuation
COPY api api
#COPY models models

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
