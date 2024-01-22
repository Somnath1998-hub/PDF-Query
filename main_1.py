# Import all necessary packages

from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import uuid
from io import BytesIO
import PyPDF2
from tqdm import tqdm
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.retrievers.document_compressors import CohereRerank
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.retrievers import ContextualCompressionRetriever
import chainlit as cl
from langchain import HuggingFaceHub
from chromadb.config import Settings
import shutil
import os

# load API KEY

hf_token = os.env['huggingfacehub_api_token']
cohere_token = os.env['cohere_api_key']

#database path
db_path = "/home/ec2-user/pdf_upload/vector_data/"

# set client and reset DB settings
client = chromadb.PersistentClient(path=db_path ,settings=Settings(allow_reset=True))
client.reset()

# load pre-trained Embeddings from HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cpu'})

# initiate text splitter for tokenization
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                                chunk_overlap=32)

# q_prompt to generate question from user query
q_prompt = """
            From given question create 4 diffrent question. Question should have the same meaning.
            Dont use your own training knowledge to create a question.
            Only give Question in return nothing else
            Question : {question}
        """


set_q_prompt = PromptTemplate(template=q_prompt,
                            input_variables=['question'])

#questions = llm(set_q_prompt.format(question=query))

# a_prompt to generate answers from all 5 queries (summarized answer)
a_prompt = """For given quesitons give a single short and precise answer from given context. Don't use your own knowledge to answer the question.
        If you dont know the answer, just say dont know. Don't try to makeup an answer
        Questions: {question}
        Context: {context}
        Helpful answer:
        """

set_a_prompt = PromptTemplate(template=a_prompt,
                            input_variables=['question',"context"])


@cl.on_chat_start
async def on_chat_start():
	"""
	On start of application clear Database and store new pdf in DB
	"""
    try:
        os.remove("/home/ec2-user/pdf_upload/vector_data/chroma.sqlite3")
        shutil.rmtree(db_path)
        print("DB clear successfully")
    except Exception as e:
        pass

    await cl.Message(content="Hello and welcome! Greetings of the day! How may I assist you today?").send()
    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your document to begin!",
            accept=["application/pdf"],
            max_size_mb=10,
            timeout=180,
        ).send()

    file = files[0]


    # Read the PDF file
    pdf_stream = BytesIO(file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()


    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Process PDF 
    docs = text_splitter.split_text(pdf_text)

    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection("my_collection")
	
    # store pdf in vector database
    for doc in tqdm(docs):
        collection.add(
            ids=[str(uuid.uuid1())], documents=doc)


    db = Chroma(client=client, collection_name="my_collection", embedding_function=embeddings)
	
    # CohereRerank to rerank the best search found
    rerank_retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 25})
    compressor = CohereRerank(user_agent = 'LLM',cohere_api_key = cohere_token)


    #compressor = CohereRerank()
    compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=rerank_retriever
        )
	
    # load llm model from HuggingFaceHub
    llm  = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta",
                      model_kwargs={"temprature":0.1,
                                    "max_new_tokens":2048,
                                    "context_length":1024},
			            huggingfacehub_api_token=hf_token)

    # RetrievalQA chain with CohereRerank 
    chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=compression_retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': set_a_prompt}
                                       )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message:str):

    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # load llm model from HuggingFaceHub
    llm  = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta",
                      model_kwargs={"temprature":0.1,
                                    "max_new_tokens":2048,
                                    "context_length":1024},
                      huggingfacehub_api_token=hf_token)

    # create 4 queries from user query
    questions = llm(set_q_prompt.format(question=message))

    #get final answer from llm 
    res = await chain.acall(message+questions, callbacks=[cb])

    answer = res["result"]
    sources = res["source_documents"]
    
    # show source document
    if sources:
        answer += f"\n\n\n\n\nSources :- " + str(sources)
                                                              
    else:
        answer += "\n\n\nNo sources found"

    await cl.Message(content=answer).send()


