from pymongo import MongoClient

client = MongoClient()
db = client.IdentiFace
dataset = db.dataset

def add(data):
    dataset.insert_one(data)

def get(filter={}, skip=0, limit=0):
    return list(dataset.find(filter, skip=skip, limit=limit))
