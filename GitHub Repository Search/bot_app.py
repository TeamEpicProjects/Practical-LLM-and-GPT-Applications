import streamlit as st

from llama_index import  load_index_from_storage, load_graph_from_storage
from llama_index import  ServiceContext, StorageContext

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig

from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine


from llama_index import GPTListIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph


import os


os.environ['OPENAI_API_KEY'] = "sk-*************************************"

def get_text():
    input_text = st.text_input("You: ","Hi, How are you?")
    return input_text


st.sidebar.title("Conversational Chat Agent")
st.title("""
ChatMate
ChatMate conversational chatterbot. Initialize the bot by clicking the "Initialize bot" button.
""")

storage_context = StorageContext.from_defaults(persist_dir=f'./storage/index')
index = load_index_from_storage(storage_context=storage_context)

index_summary = f"OpenAI Cookbook repository"

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
storage_context = StorageContext.from_defaults()

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index],
    index_summaries=[index_summary],
    service_context=service_context,
    storage_context = storage_context,
)
root_id = graph.root_id


decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
    )


query_engine = index.as_query_engine()
query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={'index_summary': index.index_struct.summary},
    )

graph_config = IndexToolConfig(
query_engine=query_engine,
name=f"Graph Index",
description="useful for when you want to answer queries that require analyzing repo.",
tool_kwargs={"return_direct": True}
)
tool_config = IndexToolConfig(
    query_engine=query_engine,
    name=f"Vector Index",
    description=f"useful for when you want to answer queries about the repo",
    tool_kwargs={"return_direct": True}
)

toolkit = LlamaToolkit(
    index_configs= [tool_config] + [graph_config],
)

memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
    )

if st.sidebar.button('Initialize bot'):
    storage_context = StorageContext.from_defaults(persist_dir=f'./storage/index')
    index = load_index_from_storage(storage_context=storage_context)

    index_summary = f"OpenAI Cookbook repository"

    # define an LLMPredictor set number of output tokens
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    storage_context = StorageContext.from_defaults()

    # define a list index over the vector indices
    # allows us to synthesize information across each index
    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [index],
        index_summaries=[index_summary],
        service_context=service_context,
        storage_context = storage_context,
    )
    root_id = graph.root_id


    decompose_transform = DecomposeQueryTransform(
        llm_predictor, verbose=True
        )


    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
            query_engine,
            query_transform=decompose_transform,
            transform_extra_info={'index_summary': index.index_struct.summary},
        )

    graph_config = IndexToolConfig(
    query_engine=query_engine,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing repo.",
    tool_kwargs={"return_direct": True}
    )
    tool_config = IndexToolConfig(
        query_engine=query_engine,
        name=f"Vector Index",
        description=f"useful for when you want to answer queries about the repo",
        tool_kwargs={"return_direct": True}
    )

    toolkit = LlamaToolkit(
        index_configs= [tool_config] + [graph_config],
    )

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm=OpenAI(temperature=0)
    agent_chain = create_llama_chat_agent(
        toolkit,
        llm,
        memory=memory,
        verbose=True
        )

    st.title("Your bot is ready to talk to you")


    user_input = get_text()
    if True:
        st.text_area("Bot:", value=agent_chain.run(input=user_input), height=200, max_chars=None, key=None)
    else:
        st.text_area("Bot:", value="Please start the bot by clicking sidebar button", height=200, max_chars=None, key=None)
