import streamlit as st
import pandas as pd
import numpy as np
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import openai
from openai.embeddings_utils import get_embedding
openai.api_key = 'sk-******************************'

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False





df = pd.read_excel('BTS_chapter2.xlsx')
#df['Number of tracks'] = df['Number of tracks'].astype(str)
#df['text'] = "Album Name: " + df['Album Name'] + "; Artist: " + df['Artist'] + "; Release Date: " + df["Release Date"] + "; Language: " + df['Language'] + "; Number of tracks: " + df['Number of tracks'] + "; Length: " + df["Length"] + "; Overview: " + df['Overview'] + "; List of song names: " + df['List of song names']
df['embedding'] = df.text.apply(lambda x: get_embedding(x, engine=f'text-embedding-ada-002'))
#df.drop(['Album Name', 'Artist', 'Language', 'Release Date', 'Number of tracks', 'Length', 'Overview', 'List of song names'], axis=1, inplace=True)


st.header("BTS since 2022")
question = st.text_input("Ask me anything about Chapter 2 projects of BTS!! ", label_visibility=st.session_state.visibility, disabled=st.session_state.disabled,)



def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]
    

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below descriptions of BTS Chapter-2 Project to answer the subsequent question. If the answer cannot be found in the description, write "I could not find an answer. I specialize in Chapter-2 of BTS and not on the topic you asked about."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_album = f'\n\nBTS Chapter-2 projects description:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_album + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_album
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about BTS Chapter-2 projects."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


if question:
    response = ask(question)
    st.write(response)


