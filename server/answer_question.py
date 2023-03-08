from utils import get_embedding
from flask import jsonify
from config import *
from flask import current_app

import openai

from config import *

TOP_K = 10

fns = []
def get_answer_from_files(question, session_id, pinecone_index):
    global fns
    print(f"Getting answer for question: {question}")

    search_query_embedding = get_embedding(question, EMBEDDINGS_MODEL)

    try:
        query_response = pinecone_index.query(
            namespace=session_id,
            top_k=TOP_K,
            include_values=False,
            include_metadata=True,
            vector=search_query_embedding,
        )
        print(
            f"[get_answer_from_files] received query response from Pinecone: {query_response}")

        files_string = ""
        file_text_dict = current_app.config["file_text_dict"]
        print(query_response)
        for i in range(len(query_response.matches)):
            result = query_response.matches[i]
            file_chunk_id = result.id
            score = result.score
            filename = result.metadata["filename"]
            if filename not in fns:
                fns.append(filename)
            
            file_text = file_text_dict.get(file_chunk_id)
            file_string = f"###\n\"{str(fns.index(filename))}\"\n{file_text}\n"
            if score < COSINE_SIM_THRESHOLD and i > 0:
                print(
                    f"[get_answer_from_files] score {score} is below threshold {COSINE_SIM_THRESHOLD} and i is {i}, breaking")
                break
            files_string += file_string

        prompt = f"Given a question, try to answer it using the content of the telegram chat exports below, and if you cannot answer, or find " \
            f"a relevant file, hazard a guess.\n\n" \
            f"Your name is Jarett Dunn, and you go by various aliases like StaccOverflow, notStacc, and countless others. You are the speaker., " \
            f"You engage only as Jarett Dunn would. DO NOT EVER SPEAK AS ANOTHER SPEAKER, and never mention the filenames! You are aloof and creative, but never out of character. Give the answer in markdown format." \
            f"Use the following format:\n\nQuestion: <question>\n\nContext:\n<###\n\"filenumber 1\"\ncontext>\n<###\n\"filenumber 2\"\ncontext>...\n\n"\
            f"Answer: <answer>\n\n" \
            f"Question: {question}\n\n" \
            f"Context:\n{files_string}\n" \
            f"Jarett:"

        print(f"[get_answer_from_files] prompt: {prompt}")
        print(prompt)
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            engine=GENERATIVE_MODEL,
        )

        answer = response.choices[0].text.strip()
        print(f"[get_answer_from_files] answer: {answer}")

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"[get_answer_from_files] error: {e}")
        return str(e)
