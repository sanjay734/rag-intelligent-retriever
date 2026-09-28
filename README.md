# Chat With Your Notes

A small RAG app I built to chat with my own study material. You upload PDFs, Word files, CSVs or text files, ask a question in normal English, and it answers using only what is written in those files.

The whole thing runs on your own laptop. There is no API key, no paid service, and nothing gets uploaded anywhere. The language model and the embedding model both run inside the Python app itself.

## Why I built this

I had a lot of notes and PDFs and Ctrl+F was not enough. It only works if you remember the exact word used. Normal chatbots don't know what is in my files, and when they don't know, they sometimes just make something up.

Most RAG tutorials I found also depend on the OpenAI API, which costs money per question and sends your documents to someone else's server. I wanted a version that is completely free and keeps everything on my machine, even if the answers are a bit less polished.

## Screenshots

Home screen:



![Home](docs/screenshots/01-home.png)



After uploading and processing a document:



![Upload](docs/screenshots/02-upload.png)



An answer with the source text it was based on:



![Answer](docs/screenshots/03-answer.png)



Chat history:



![History](docs/screenshots/04-history.png)



## What it can do

- Upload PDF, DOCX, CSV and TXT files, several at once
- Read tables inside PDFs and Word files, not just plain text
- Split documents into overlapping chunks so sentences don't get cut off badly
- Find relevant text by meaning, so a question like "how do plants get energy" can match a paragraph about photosynthesis even without the same words
- Show the source chunks (file name, page number, text) under every answer so you can check it yourself
- Keep a short chat history for the session
- Clear all loaded documents or just the chat with one button
- Work offline once the models have been downloaded

## Tech stack

- Python 3.11
- Streamlit for the web interface
- LangChain to connect the loading, retrieval and answering steps
- sentence-transformers (all-MiniLM-L6-v2) for embeddings, 384 dimensions, runs on CPU
- HuggingFace Transformers with SmolLM2-360M-Instruct as the model that writes answers
- FAISS as the vector store
- pdfplumber and pypdf for PDFs (pypdf is the fallback if pdfplumber fails)
- python-docx for Word files
- the csv module for CSV files

## How it works

1. The uploaded file is read and the text is pulled out. PDFs are read page by page so the page number can be kept.
2. The text is cut into chunks of about 1000 characters with a 200 character overlap. The code tries to cut at the end of a sentence or a line break instead of the middle of a word.
3. Each chunk is turned into a list of numbers (an embedding) that represents its meaning.
4. The embeddings are stored in a FAISS index in memory.
5. When you ask a question, the question is turned into an embedding the same way, and FAISS returns the 5 closest chunks.
6. Those 5 chunks and the question go into a prompt for the local model. The prompt tells it to answer only from the given text and to say so if the answer is not there.
7. The answer is shown along with the chunks it came from.

The easiest way I explain it is an open book exam. The model does not memorise your file. It first looks up the right page and then answers from that page.

## Project structure

```
rag_assistant_free/
    app.py                  Streamlit interface
    rag_system.py           embeddings, FAISS, retrieval, answer generation
    document_processor.py   reading files and chunking
    config.py               model names and settings
    requirements.txt
    .env.example
    docs/screenshots/       images used in this README
    data/vector_store/      optional place to save the FAISS index
```

## Setup

You need Python and around 4 GB of free RAM. Internet is only needed for the first run, because two models get downloaded (roughly 800 MB in total).

On Windows, in Command Prompt:

```
cd rag_assistant_free
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On Mac or Linux the activate line is `source venv/bin/activate`.

The app opens at http://localhost:8501.

The first run is slow because it downloads the models. The browser only shows a spinner during this. The real progress bars are in the terminal, so look there if it seems stuck.

To run it again on another day you do not need to install anything again. Just do this:

```
cd rag_assistant_free
venv\Scripts\activate
streamlit run app.py
```

## How to use it

1. Upload one or more files from the sidebar.
2. Click Process Documents and wait until the chunk count shows up.
3. Type a question and click Search.
4. Open the source boxes under the answer to see which text it used.

Some questions I tried: "Summarize the main points", "What does the document say about polymorphism", "List all the dates mentioned".

## Settings

Everything is in `config.py`.

- CHUNK_SIZE = 1000
- CHUNK_OVERLAP = 200
- TOP_K_RETRIEVAL = 5
- GENERATION_TEMPERATURE = 0.3 (kept low so answers stay close to the document)
- EMBEDDING_MODEL = sentence-transformers/all-MiniLM-L6-v2
- LOCAL_LLM_MODEL = HuggingFaceTB/SmolLM2-360M-Instruct

If your internet is slow or your RAM is low, use a smaller model by changing `LOCAL_LLM_MODEL`:

- HuggingFaceTB/SmolLM2-135M-Instruct, about 270 MB, basic answers
- HuggingFaceTB/SmolLM2-360M-Instruct, about 750 MB, the default
- Qwen/Qwen2.5-0.5B-Instruct, about 1 GB, better answers

## Problems I ran into

**Chunking loop that never ended.** My first chunking function got stuck forever on some inputs. When the last piece of text was short, subtracting the overlap moved the starting point backwards, so the loop kept repeating the same slice. I fixed it by stopping as soon as the final chunk is reached and by making sure the start position always moves forward.

**Ollama was too heavy.** My first free version used Ollama with Llama 3.2. That meant installing a separate program and downloading about 2 GB, which was too much for my internet. I switched to running a small model directly through Transformers, so there is nothing extra to install.

**Streamlit error from a one line if/else.** I had written an `st.success(...) if ... else st.info(...)` expression on its own line, and Streamlit tried to display its return value and crashed. Changing it to a normal if/else block fixed it.

## Limitations

- The model is very small, so answers are simpler and less reliable than what you get from big models like GPT-4. It is fine for finding and summarising things from your files, but it will struggle with complicated reasoning.
- There is no conversation memory yet. Every question is treated on its own, so follow up questions like "explain that more" will not work.
- Chunking is based on character count, not on headings or sections.
- Speed depends on your computer. On a CPU only laptop, answers can take a while.
- I have not measured answer accuracy in any formal way. I only checked by reading the answers against the source text.

## Things I want to add

- Conversation memory for follow up questions
- Chunking that follows headings and sections
- Keyword search combined with semantic search
- PPTX and HTML support
- A simple script to check how good the retrieval is

## Author

Sanjay P
LinkedIn: (https://www.linkedin.com/in/sanjay-p-715939240)
