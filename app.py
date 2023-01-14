from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel
import requests

nlp = pipeline(task='sentiment-analysis',
               model='cardiffnlp/twitter-roberta-base-sentiment')

app = FastAPI()

class Request(BaseModel):
    text: str

@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}


@app.post('/sentiment_analysis/')
async def query_sentiment_analysis(text: Request):
    return analyze_sentiment(text)


def analyze_sentiment(request):
    """Get and process result"""

    print(request);
    result = nlp(request.text)
    print(result);

    sent = ''
    if (result[0]['label'] == 'LABEL_0'):
        sent = 'Negative'
    elif (result[0]['label'] == 'LABEL_1'):
        sent = 'Neutral'
    elif (result[0]['label'] == 'LABEL_2'):
        sent = 'Positive'

    prob = round(result[0]['score'], 2)
    print(prob)

    # Format and return results
    return {'sentiment': sent, 'probability': round(prob,2)}