# 🧠 Bilingual Medical Question-Answering System (MQAS)

This repository contains a context-aware, bilingual (Arabic-English) medical question-answering system using ensemble learning with GPT-Neo, RoBERTa, and Knowledge-Aware Neural Networks (KANs). The final answer is selected via a **weighted majority voting mechanism** based on model confidence.

---

## 🚀 Features

- 🔄 Bilingual support for Arabic and English queries
- 🤖 Multi-model response generation using:
  - GPT-Neo (Generative model)
  - RoBERTa (Discriminative model)
  - KANs (Knowledge-aware model)
- 📊 Weighted majority voting for ensemble decision making
- 💡 Semantic similarity-based answer grouping
- ⚙️ Modular design for easy extension and deployment

---

## 📂 Project Structure

```
bilingual-mqas/
│
├── models/
│   ├── gpt_neo_model.py         # GPT-Neo answer generation
│   ├── roberta_model.py         # RoBERTa answer extraction
│   └── kan_model.py             # Knowledge-Aware Neural Network response
│
├── utils/
│   ├── preprocessing.py         # Query normalization and cleaning
│   ├── language_detector.py     # Language detection using langdetect
│   └── confidence_scorer.py     # Heuristic-based model confidence scoring
│
├── ensemble/
│   └── weighted_majority_voting.py # Voting logic with semantic similarity
│
├── main.py                      # Entry point to run the system
├── requirements.txt             # Python dependencies (to be created)
└── README.md                    # This file
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bilingual-mqas.git
cd bilingual-mqas

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

---

## 🧪 Usage

```bash
python main.py
```

You'll be prompted to enter a medical question in either Arabic or English. The system will return the best answer based on combined model responses.

---

## 📚 Dependencies

To be included in `requirements.txt`, includes:
- `transformers`
- `torch`
- `sentence-transformers`
- `scikit-learn`
- `langdetect`

---

## 📌 Notes

- Current KAN implementation is a placeholder. You can replace it with a real knowledge graph-enhanced transformer.
- RoBERTa context is mock-filled — replace it with your actual medical corpora or document index.

---

## 👨‍⚕️ Potential Applications

- Telemedicine chatbots  
- Patient-facing hospital kiosks  
- Digital health assistants  
- Medical education platforms  

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

## 🙌 Acknowledgements

- HuggingFace Transformers  
- Sentence Transformers  
- Langdetect  
- scikit-learn