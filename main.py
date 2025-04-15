from models.gpt_neo_model import generate_gpt_response
from models.roberta_model import generate_roberta_response
from models.kan_model import generate_kan_response

from utils.preprocessing import preprocess_query
from utils.language_detector import detect_language
from utils.confidence_scorer import get_confidence_score
from ensemble.weighted_majority_voting import weighted_vote

def run_mqas_pipeline(query):
    lang = detect_language(query)
    query = preprocess_query(query)

    gpt_answer = generate_gpt_response(query)
    roberta_answer = generate_roberta_response(query, lang)
    kan_answer = generate_kan_response(query, lang)

    w_gpt = get_confidence_score(gpt_answer)
    w_roberta = get_confidence_score(roberta_answer)
    w_kan = get_confidence_score(kan_answer)

    final_answer = weighted_vote(
        answers=[gpt_answer, roberta_answer, kan_answer],
        weights=[w_gpt, w_roberta, w_kan]
    )

    return final_answer

if __name__ == "__main__":
    user_query = input("Enter your medical query: ")
    print("Answer:", run_mqas_pipeline(user_query))