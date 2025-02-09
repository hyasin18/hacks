from flask import Flask, request, jsonify, render_template 
import json
import os
import openai
import numpy as np
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Store your API key in .env

# Load all JSON files into a single database list
database = []
data_folder = "data"

# Ensure data folder exists
if not os.path.exists(data_folder):
    print(f"⚠️ Data folder '{data_folder}' not found!")
else:
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        print(f"✅ Loaded {len(data)} questions from {filename}")
                        database.extend(data)
                    else:
                        print(f"⚠️ Invalid format in {filename}. Expected a list of Q&A.")
            except json.JSONDecodeError:
                print(f"❌ Failed to parse {filename}! Check the JSON format.")

print(f"📊 Total questions loaded: {len(database)}")


# Find the best answer using fuzzy matching
def find_best_answer(user_question):
    user_question = user_question.lower()

    # Debugging: Check if database is empty
    if not database:
        print("⚠️ Database is empty! Make sure JSON files are loaded properly.")
        return "I'm sorry, no data available."

    questions = [entry["question"].lower() for entry in database]
    print(f"🔍 User question: {user_question}")
    print(f"📂 Total questions in database: {len(questions)}")

    # Try to find a match with fuzzy search
    best_match, score = process.extractOne(user_question, questions, scorer=fuzz.partial_ratio, score_cutoff=60)
    print(f"✅ Best fuzzy match found: {best_match}, Score: {score}")
    
    if best_match:
        for entry in database:
            if entry["question"].lower() == best_match:
                print("🎯 Returning database answer:", entry["answer"])
                return entry["answer"]
    
    # Refined word-matching search
    words = set(user_question.split())
    for entry in database:
        question_words = set(entry["question"].lower().split())
        common_words = words.intersection(question_words)
        if len(common_words) >= 2:  # Require at least 2 common words
            print(f"✅ Found match with words: {common_words}")
            return entry["answer"]
    
    print("❌ No match found. Calling OpenAI API...")
    return get_openai_answer(user_question)


# Get a fallback answer from OpenAI
def get_openai_answer(user_question):
    prompt = f"The following is a Q&A system based on Islamic knowledge.\n\nQuestion: {user_question}\nAnswer:"

    try:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        openai_answer = response.choices[0].text.strip()
        print(f"🤖 OpenAI Answer: {openai_answer}")
        return openai_answer
    except Exception as e:
        print(f"❌ Error with OpenAI API: {e}")
        return "I'm sorry, something went wrong. Please try again later."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json.get("question", "").strip()
    print(f"📥 Received question: {user_question}")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    answer = find_best_answer(user_question)
    print(f"📤 Answer sent: {answer}") 

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)