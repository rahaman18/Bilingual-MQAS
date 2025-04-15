from langdetect import detect

def detect_language(query):
    return detect(query)