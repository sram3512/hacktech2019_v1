# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
from flask import Flask
from flask import request
from flask import Response
from flask import jsonify
from flask_cors import CORS
app=Flask(__name__)
CORS(app)

con=1

@app.route('/userquery',methods=['POST'])
def query():
    global con
    if request.is_json:
        content = request.get_json()
        print (content['user_query'])
        user_input=content['user_query']
	responsetuple=()
    	usernp=""
        if con!=-1:
            responsetuple = botresponse(user_input,con)
            print (responsetuple)
	    for element in responsetuple[1]:
		usernp+=str(element.encode("utf-8")) 
		usernp+=";" 
            con = responsetuple[0]
          
        else:
            con=0
            print ("Completed Run")

	dtval=""
	dtval+=(usernp[:-1]+","+responsetuple[2]+","+str(responsetuple[0]))
	return Response(dtval,status=200,mimetype='application/json')
    else:
        print ("Not in json")
        return Response(str('{"message":"Input must be json"}'),status=400,mimetype='application/json')



def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def classify(sentence,ERROR_THRESHOLD):
    # generate probabilities from the model

    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def botresponse(sentence,con):
    botresponsemsg = ""
    results = ()
    ERROR_THRESHOLD = 0.25
    while True:
        results = classify(sentence,ERROR_THRESHOLD)
        ERROR_THRESHOLD = ERROR_THRESHOLD - 0.1
        for result in results:
            for i in intents['intents']:
                if i['tag'] == result[0]:
                    if i['context_filter'] == con:
                        botresponsemsg = random.choice(i['responses'])
                        con = int(i['context_set'])
                        nextpatterns = []
                        for j in intents['intents']:
                            if int(j['context_filter']) == int(i['context_set']):
                                for p in j['patterns']:
                                    nextpatterns.append(p)
                        return ((con, nextpatterns, botresponsemsg))
        if ERROR_THRESHOLD != 0:
            continue
    for result in results:
        for i in intents['intents']:
            if i['tag'] == result[0]:
                botresponsemsg = random.choice(i['responses'])
                con = int(i['context_set'])
                nextpatterns = []
                for j in intents['intents']:
                    if int(j['context_filter']) == int(i['context_set']):
                        for p in j['patterns']:
                            nextpatterns.append(p)
                return ((con, nextpatterns, botresponsemsg))

if __name__=='__main__':

    with open('intents.json') as json_data:
        intents = json.load(json_data)

    nltk.download('punkt')
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    training = []
    output = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:,0])
    train_y = list(training[:,1])

    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(train_x[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
    model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')
    model.load('./model.tflearn')
  
    app.run(debug=True,host='0.0.0.0')
