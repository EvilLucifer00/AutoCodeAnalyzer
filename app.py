from src.helper import load_embedding, repo_ingestion
from dotenv import load_dotenv
import os
from flask import Flask, render_template, jsonify, request
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
# Updated imports for modern LangChain
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

# Load environment variables from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize embeddings and vector database
embeddings = load_embedding()
persist_directory = "db"

# Load the persisted database from disk
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Setup LLM with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# Setup Memory and Retrieval Chain using modern LangChain syntax
memory = ConversationSummaryMemory(
    llm=llm, 
    memory_key="chat_history", 
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
    memory=memory
)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():
    user_input = ""
    if request.method == 'POST':
        user_input = request.form.get('question', '')
        if user_input:
            repo_ingestion(user_input)
            # This triggers the indexing script to process the newly cloned repo
            os.system("python store_index.py")

    return jsonify({"response": str(user_input)})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("User Input:", input)
    
    try:
        # Try to get the answer from Google
        result = qa.invoke({"question": msg})
        print("Response:", result["answer"])
        return str(result["answer"])
        
    except Exception as e:
        error_message = str(e)
        # Catch the rate limit crash specifically
        if "429" in error_message or "Quota" in error_message:
            print("API RATE LIMIT HIT")
            return "⏳ <b>Google API Rate Limit Hit!</b><br>The free tier allows roughly 2 questions per minute. Please wait 60 seconds and click send ONCE."
        
        # Catch any other random crashes so the server doesn't die
        print(f"Backend Error: {error_message}")
        return f"⚠️ An internal error occurred. Check the terminal for details."

if __name__ == '__main__':
    # Using 8080 as per your original configuration
    app.run(host="0.0.0.0", port=8080, debug=False)