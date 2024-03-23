from django.shortcuts import render
from django.http import HttpResponse
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


def index(request):
    if request.method == "POST":
        get_text = request.POST['texto']
        print(get_text)
        polar_text = polarize(get_text)
        return render(request, "polarity.html",{'text':polar_text})
    else:
        return render(request, "index.html")


def polarize(text):
    tokens = word_tokenize(text)
    print(tokens)
    senti_val = [get_sentiment(x,tokens) for (x) in tokens]
    print(senti_val)
    return senti_val

def get_sentiment(word, tokens):
    synset = lesk(tokens,word)
    print(word)
    print(synset)
    if synset:
        swn_synset = swn.senti_synset(synset.name())
        return [word,swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.pos_score()-swn_synset.neg_score()]
    else:
        return [word,0.0,0.0,0.0]
