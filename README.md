Task 4 — RAG-Powered Conversational AI (Gemini API)
Overview

This web application is a RAG-powered chatbot that retrieves relevant facts from your uploaded documents and generates human-like answers using the Google Gemini API.

RAG stands for Retrieval-Augmented Generation:

First, the app retrieves context from your uploaded documents.

Then, the LLM (Gemini API) uses this context to generate polished, context-aware responses.

This approach ensures more accurate and informative answers compared to standard chatbots that rely solely on the model.

Features

✅ Upload TXT/PDF files to create a knowledge base.

✅ Keyword-based semantic search for retrieving relevant context.

✅ Integration with Google Gemini API for natural language generation.

✅ Interactive chat interface built with Streamlit.

✅ Buttons to Clear Chat and Clear Knowledge Base.

✅ Fully task-specific, no extra features.

How it Works

Upload Documents:

Upload TXT or PDF files via the sidebar.

Click Add files to build your knowledge base.

Ask a Question:

Type your question in the chat input and click Ask ✨.

Retrieve Context:

The app performs a keyword-based search in the knowledge base to find relevant context.

Generate Answer:

The question and retrieved context are sent to the Gemini API prompt.

Gemini LLM generates a clear, human-like, context-aware answer.

View Response:

The chat history displays your questions and the AI’s responses.

Benefits

⚡ Accurate Answers: Responses are based only on the uploaded documents.

🧠 Human-like Answers: Gemini API generates natural, readable responses.

📚 Custom Knowledge Base: Answers are tailored to your own data.

🔄 Interactive Chat: Ask multiple questions in real-time with a simple web interface.

Setup & Installation

Clone Repository

```bash
git clone https://github.com/sundasmustaf69/Task4.git
cd Task4

Install Dependencies

pip install streamlit google-generativeai pypdf


Run Streamlit App

streamlit run task4_streamlit_gemini.py


Provide Gemini API Key

Enter your API key in the sidebar (Google AI Studio > Gemini API).

Select a model: gemini-1.5-flash or gemini-1.5-pro.

Upload Knowledge Base

Upload TXT or PDF files in the sidebar.

Click Add files to build the knowledge base.

Start Chatting

Type your questions aur answers ke liye simple web interface.
