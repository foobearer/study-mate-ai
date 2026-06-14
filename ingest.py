from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import (ChatOpenAI, OpenAIEmbeddings)

from langchain_chroma.vectorstores import Chroma

# from prompts import PROMPT_RETRIEVING_HUM, PROMPT_RETRIEVING_SYS

# this loads the files and returns as a list of documents
print("📄 Loading PDF...")
loader_pdf = PyPDFLoader("intro-to-ai-transcript.pdf")
docs_list = loader_pdf.load()
print(f"✓ Loaded {len(docs_list)} documents")

# Chunking
# splits the documents to make sure it does not contain more than 200 tokens 
# this will feed the chatbot the shortest text possible during q&A stage to optimize cost.
print("\n✂️  Splitting documents into chunks...")
token_splitter = TokenTextSplitter(encoding_name="cl100k_base",
                                   chunk_size=200,
                                   chunk_overlap=40)

docs_list_tokens_split = token_splitter.split_documents(docs_list)
print(f"✓ Created {len(docs_list_tokens_split)} chunks")

# Vectorization
# to transform each document into an array of numbers, also known as vectors or embeddings
# this later would allow the chatbot to quickly find the text most relevant to the question
print("\n🔢 Creating embeddings...")
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# this keeps vectors to a local DB
print("💾 Storing vectors in Chroma...")
vectorstore = Chroma.from_documents(documents=docs_list_tokens_split,
                                    embedding=embedding,
                                    persist_directory='./vectorised-data')
print("✓ Vector store created")


#Retrieval
#retriever's goal is to find the most relevant document to the question's asked
print("\n🔍 Testing retriever...")
retriever = vectorstore.as_retriever(search_type='mmr',
                                     search_kwargs={'k':1, 'lambda_mult':0.7}) 

# result = retriever.invoke("What is Newural networks")
# print(f"✓ Query result:\n{result[0].page_content}")
print("\n✅ Ingest complete!")

#Prompting
# this would instruct the model of its intended behaviour and purpose
PROMPT_RETRIEVING_SYS = '''You will receive a question from a student taking the Intro to AI course.
    Answer the question using only the provided context.
'''

PROMPT_RETRIEVING_HUM = '''his is the question:
{question}

This is the context:
{context}

'''

prompt_retrieving_sys = SystemMessage(PROMPT_RETRIEVING_SYS)
prompt_retrieving_hum = HumanMessagePromptTemplate.from_template(PROMPT_RETRIEVING_HUM)

chat_prompt_retrieving_template = ChatPromptTemplate([PROMPT_RETRIEVING_SYS, PROMPT_RETRIEVING_HUM])
print("\n✅ Prompting complete!", chat_prompt_retrieving_template)

#LLM 
# temperature 0 - to ensure it will have same answers each time
chat = ChatOpenAI(model_name='gpt-4o', temperature=0)

print("\n✅ LLM complete!", chat)


#Parser
# string output parser - ensures the result will be a string
str_output_parser = StrOutputParser()

#Chaining
#chain contruction link components by passing the output of one as an argument to the next.
#this basically stores questions and context in a dictionary
chain = ({
    'context': retriever,
    'question': RunnablePassthrough()} #store Q&C in a dictionary
    | chat_prompt_retrieving_template #feed the dict to the prompt template
    | chat #pass the resulting promt to the LLM
    | str_output_parser #feed the model's response tot he parser
)


response = chain.stream("When did AI programs demonstrated that machines could perform symbolic reasoning?")
#We have to loop through each chunk 
print("\n RESPONSE:")
for chunk in response:
    print(chunk, end="")

