📝 Resume Screening Using Machine Learning

Automatically screens resumes based on skills, experience, and job description matching using NLP & ML.

🚀 Project Overview

This project extracts text from resumes, processes it using Natural Language Processing (NLP), and predicts whether a candidate is suitable for a job role.
It helps automate the hiring process by saving time and improving accuracy.

🎯 Features

📄 Resume text extraction

🔤 NLP-based preprocessing

🤖 Machine learning classification

📊 Candidate score calculation

🎯 Job description matching

🖥️ Web interface for uploading resumes (if included)

🧰 Tech Stack

Component	Technology
Programming Language	Python
NLP	NLTK / SpaCy
ML Models	Logistic Regression / SVM / Random Forest
Vectorization	TF-IDF
Web Framework (optional)	Flask / Streamlit


📁 Project Structure

Resume-Screening/
│── data/
│   ├── resumes/
│   ├── job_descriptions/
│── models/
│── notebooks/
│── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── predict.py
│── app.py  (if using Flask)
│── requirements.txt
│── README.md

🔧 How to Run the Project

1️⃣ Clone the repository
git clone https://github.com/HarshaliItkar/Resume-Screening.git
cd Resume-Screening

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the model or web app

If you're running a script:

python main.py


If using Flask:

python app.py

📊 Model Workflow

Extract text from resume (PDF/docx)

Clean and preprocess text (stopwords, stemming, lowercasing)

Convert text → numerical features (TF-IDF)

Predict candidate suitability

Generate a match score

📝 Outputs

Candidate suitability: Selected / Rejected

Match percentage

Extracted skills

Key missing skills

Prediction confidence

🚀 Future Enhancements

Add deep learning models (BERT)

ATS-style ranking

Multi-role resume matching

Deploy as web app using AWS / Render

🤝 Contributing

Feel free to fork this project, make improvements, and create a pull request!

📬 Contact

👤 Harshali Itkar
🔗 GitHub: HarshaliItkar
📧 Email: harshaliitkar2211@gmail.com
