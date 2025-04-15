from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def weighted_vote(answers, weights):
    grouped_answers = defaultdict(float)
    for i, ans1 in enumerate(answers):
        embedding1 = model.encode([ans1])
        matched = False
        for group in grouped_answers:
            embedding2 = model.encode([group])
            sim = cosine_similarity(embedding1, embedding2)[0][0]
            if sim > 0.8:
                grouped_answers[group] += weights[i]
                matched = True
                break
        if not matched:
            grouped_answers[ans1] += weights[i]
    return max(grouped_answers, key=grouped_answers.get)