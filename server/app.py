from __future__ import print_function
from config import *

import tiktoken
import pinecone
import uuid
import sys
import logging

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from flask import request

from handle_file import handle_file
from answer_question import get_answer_from_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

import time
def get_embeddings(text_array, engine):
    # Parameters for exponential backoff
    max_retries = 5 # Maximum number of retries
    base_delay = 1 # Base delay in seconds
    factor = 2 # Factor to multiply the delay by after each retry
    while True:
        try:
            return openai.Engine(id=engine).embeddings(input=text_array)["data"]
        except Exception as e:
            if max_retries > 0:
                time.sleep(base_delay)
                max_retries -= 1
                base_delay *= factor
            else:
                raise e
def load_pinecone_index() -> pinecone.Index:
    """
    Load index from Pinecone, raise error if the index can't be found.
    """
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = PINECONE_INDEX
    
    index = pinecone.Index(index_name)

    return index
from os.path import isfile, join
from os import listdir
import pandas as pd 
from openai.embeddings_utils import cosine_similarity
embedding_cache_path = "C:\\temp\\70devs.csv"
default_embedding_engine = "text-embedding-ada-002" 
import pickle 
from app import load_pinecone_index

from utils import get_pinecone_id_for_file_chunk

import openai 
iii = -1

#df = pd.read_csv(embedding_cache_path, index_col=False)
#df.drop_duplicates(subset=['completion_id'], inplace=True)
#print(len(df))
pinecone_index = load_pinecone_index()
#embedding_cache = {( d.texts, default_embedding_engine): d.embedding for d in pd.read_csv(embedding_cache_path).itertuples()}
texts = []
vectors = []
doing = False
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j

def get_embedding_with_cache(pretext, filename, engine=default_embedding_engine):

    global df,embedding_cache, embedding_cache_path, iii, pinecone_index, texts, vectors, doing 
    print(filename)
    try:

        """Handle a file string by creating embeddings and upserting them to Pinecone."""
        logging.info("[handle_file_string] Starting...")

        # Clean up the file string by replacing newlines and double spaces
        clean_file_body_string = pretext.replace(
            "\n", "; ").replace("  ", " ")
        # Add the filename to the text to embed
        text_to_embed = "Filename is: {}; {}".format(
            filename, clean_file_body_string)

        # Create embeddings for the text
        try:
            token_chunks = list(chunks(clean_file_body_string, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
            text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]
            for i in range(1,len(text_chunks)):
                id = get_pinecone_id_for_file_chunk("jarett", filename, i)
                print(id)
                check = pinecone_index.fetch([id], namespace="jarett")
                if len(check.vectors) > 0:
                    doing = True
                    print("not doing")
                    text_embeddings = check.vectors
                    break
                # Split text_chunks into shorter arrays of max length 10
                text_chunks_arrays = text_chunks[i:i+MAX_TEXTS_TO_EMBED_BATCH_SIZE] 

                # Call get_embeddings for each shorter array and combine the results
                embeddings = []
                for text_chunks_array in text_chunks_arrays:
                    embeddings_response = get_embeddings(text_chunks_array, engine)
                    embeddings.extend([embedding for embedding in embeddings_response])

                text_embeddings = list(zip(text_chunks, embeddings))
                embedding_cache[(text_chunks, engine)] = text_embeddings
                logging.info(
                    "[handle_file_string] Created embedding for {}".format(filename))
                
                # Get the vectors array of triples: file_chunk_id, embedding, metadata for each embedding
                # Metadata is a dict with keys: filename, file_chunk_index
                id = get_pinecone_id_for_file_chunk("jarett", filename, i)
                vectors.append(
                    (id, text_embeddings, {"filename": filename, "file_chunk_index": i}))

                try:
                    pinecone_index.upsert(
                        vectors=vectors, namespace="jarett")

                    logging.info(
                        "[handle_file_string] Upserted batch of embeddings for {}".format(filename))
                except Exception as e:
                    logging.error(
                        "[handle_file_string] Error upserting batch of embeddings to Pinecone: {}".format(e))
                    raise e
        except Exception as e:
            logging.error(
                "[handle_file_string] Error creating embeddings: {}".format(e))
            raise e
    except Exception as e:
        logging.error(
            "[handle_file_string] Error creating embeddings: {}".format(e))
        raise e
    try:
        embedding_cache[(pretext, engine)] = text_embeddings
    except:
        embedding_cache = {}
        embedding_cache[(pretext, engine)] = text_embeddings
    return embedding_cache[(pretext, engine)]
    
from os.path import isfile, join
from os import listdir
FILES_DIR = "C:\\Users\\jared\\Downloads\\tg"
def create_app():
    pinecone_index = load_pinecone_index()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    app = Flask(__name__)
    app.pinecone_index = pinecone_index
    app.tokenizer = tokenizer
    app.session_id = "jarett"
    # log session id
    app.config["file_text_dict"] = {}
    # this function will get embeddings from the cache and save them there afterward
    

    app.config["file_text_dict"] = {}
    for file in listdir(FILES_DIR):
        if 'txt' in file:
            with open(FILES_DIR + "/" + file, 'r') as f:
                app.config["file_text_dict"][file] = f.read()
    #for file in app.config["file_text_dict"]:
    #    get_embedding_with_cache(app.config["file_text_dict"][file], file)
    CORS(app, supports_credentials=True)

    return app

    
    #df = df[df['likes'] >= average_likes / 2]
    print(df.head())

    # this function will get embeddings from the cache and save them there afterward
   
    app.config["file_text_dict"] = {embedding_cache_path: df.texts.tolist()}
    CORS(app, supports_credentials=True)

    return app

app = create_app()

@app.route(f"/process_file", methods=["POST"])
@cross_origin(supports_credentials=True)
def process_file():
    try:
        file = request.files['file']
        logging.info(str(file))
        handle_file(
            file, app.session_id, app.pinecone_index, app.tokenizer)
        return jsonify({"success": True})
    except Exception as e:
        logging.error(str(e))
        return jsonify({"success": False})

@app.route(f"/answer_question", methods=["POST"])
@cross_origin(supports_credentials=True)
def answer_question():
    try:
        params = request.get_json()
        question = params["question"]

        answer_question_response = get_answer_from_files(
            question, app.session_id, app.pinecone_index)
        return answer_question_response
    except Exception as e:
        return str(e)

@app.route("/healthcheck", methods=["GET"])
@cross_origin(supports_credentials=True)
def healthcheck():
    return "OK"

if __name__ == "__main__":
    app.run(debug=True, port=SERVER_PORT, threaded=True)
