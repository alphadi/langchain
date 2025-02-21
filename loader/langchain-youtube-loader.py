# Installation requirements

# pip install youtube-transcript-api
# pip install pytube
# pip install llama-index
# pip install langchain-openai
# pip install langchain-community


## First Initialize your LLM

from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


## Use youtube loader  

loader = YoutubeLoader.from_youtube_url("", add_video_info=False)
result = loader.load()


print (type(result))
print (f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")
print ("")
print(result)

##  Initialize your LLM chain

chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
chain.invoke(result)

## Use Text Splitter to break down long youtube videos

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

texts = text_splitter.split_documents(result)

chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.invoke(texts[:4])

## Try multiple videos

youtube_url_list = ["", ""]

texts = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

for url in youtube_url_list:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    result = loader.load()
    
    texts.extend(text_splitter.split_documents(result))
