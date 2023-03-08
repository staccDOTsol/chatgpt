import openai
import logging
import sys
import time
from openai.embeddings_utils import get_embedding, cosine_similarity

from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ] 
)

def get_pinecone_id_for_file_chunk(session_id, filename, chunk_index):
    return str(session_id+"-!"+filename+"-!"+str(chunk_index))


