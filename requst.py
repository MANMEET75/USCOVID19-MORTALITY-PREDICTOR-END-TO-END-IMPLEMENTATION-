import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sepal_length':5.1, 'sepal_width':3.4, 'petal_length':1.5,'petal_width':2.3})

print(r.json())