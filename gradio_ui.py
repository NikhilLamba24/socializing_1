from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import warnings
warnings.filterwarnings('ignore')
import gradio as gr
from llama_index.core import SimpleDirectoryReader

import asyncio
import concurrent.futures
import twikit

USERNAME = 'nickalodean9'
EMAIL = 'nickalodean9@gmail.com'
PASSWORD = 'Nickalodean9'

# Initialize client
client = twikit.Client('en-IN')

async def search_tweets(client, query):
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD
    )
    tweets = await client.search_tweet(query, 'Latest', count=30)
    return tweets

def run_async_task(client, query, tweets_list):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tweets = loop.run_until_complete(search_tweets(client, query))
    tweets_list.extend(tweets)
    loop.close()




def sett_product_name(var):

  queries = []#['hair product', 'skin care', 'beauty tips']
  queries.append(var)
  tweets_list = []

  with concurrent.futures.ThreadPoolExecutor() as executor:
      for query in queries:
          executor.submit(run_async_task, client, query, tweets_list)

  #print(tweets_list)
  tweets=tweets_list
  lst = [tweet.text for tweet in tweets]

  print("Tweets retrieved:")
  # for tweet in lst:
  #     print(tweet)
  text = ",".join(lst)
  file_name = "database.txt"

  # Clear the file by opening it in write mode, which overwrites any existing content
  with open(file_name, "w") as file:
      file.write("")

  # Now write the new data to the file
  with open(file_name, "w") as file:
      file.write(text)
    #return file_name
  return file_name

GROQ_API_KEY = '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'

def embedding(file_name):

    documents = SimpleDirectoryReader(
        input_files=[file_name]
    ).load_data()
    #documents = file_name
    text_splitter = SentenceSplitter(chunk_size=4000, chunk_overlap=200)
    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    #print("Document ID:", documents[0].doc_id)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context, node_parser=nodes)
    vector_index.storage_context.persist(persist_dir="./storage_mini1")
    storage_context = StorageContext.from_defaults(persist_dir="./storage_mini1")
    index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = index.as_query_engine(service_context=service_context)
    return query_engine

import gradio as gr
import asyncio

def process_query(product_name, query):
      X_X=sett_product_name(product_name)
      query_engine=embedding(X_X)
      query_s="""based on this provide me some better post that target this issues for product launch i want only post content with important #tag
      for posting on twitter (as per text constraints for thread length. and i want only one post it should be most effective and provide all the solution)
      please don't include these type of text like'Here's a potential Twitter post for launching your "LCG" hair product:'. i want to post directly on twitter
      and please make genrate the content only within 280 charecters. """
      query = query + query_s
      resp = query_engine.query(query)
      return resp

iface = gr.Interface(
    fn=process_query,
    inputs=["text", "text"],
    outputs="text",
    title="Post Generation",
    description="Generate a Twitter post for product launch based on product name and query."
)
