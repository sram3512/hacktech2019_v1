from flask import Flask
from flask import request
import json
from flask import Response
app=Flask(__name__)
@app.route('/userquery',methods=['POST'])
def query():
    if request.is_json:
        content = request.get_json()
        print (content['user_query'])
        return Response("{'message':'OK'}",status=200,mimetype='application/json')
    else:
        print ("Not in json")
        return Response("{'message':'Input must be json'}",status=400,mimetype='application/json')
    
