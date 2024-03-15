from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM,pipeline, AutoTokenizer
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 

pdfLoader = PyPDFLoader("/media/nsl3090-3/hdd1/hujaifa/Langchain_RAG/PDF/giji.pdf")
documents = pdfLoader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20, separators='\n\n\n')
docs = text_splitter.split_documents(documents)

modelPath = "intfloat/multilingual-e5-large"
model_kwargs = {'device':'cuda:0'}
encode_kwargs = {'normalize_embeddings':False}
embeddings = HuggingFaceEmbeddings(
  model_name = modelPath,  
  model_kwargs = model_kwargs,
  encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(docs, embeddings)
question = "人工知能関連の政策を議論する内閣府の「第2回AI戦略会議」の構成員は誰ですか？"
searchDocs = db.similarity_search(question)
print(searchDocs[0].page_content)

tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")
model = AutoModelForCausalLM.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
                                            #  load_in_8bit=True,
                                            device_map="auto",)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device='cuda:1')
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(
    pipeline = pipe,
    model_kwargs={"temperature": 0, "max_length": 512},
)

# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
# {context}
# Question: {question}
# Helpful Answer:"""

template = """[INST] <>
あなたは誠実で優秀な日本人のアシスタントです。
マークダウン形式で以下のコンテキスト情報を元に質問に回答してください。
<>

{context}

{question}
[/INST]"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
print('Result:::')
qa_chain = RetrievalQA.from_chain_type(   
  llm=llm,   
  chain_type="stuff",   
  retriever=db.as_retriever(),   
  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
) 
result = qa_chain ({ "query" : question })
print('Result:::') 
print(result["result"])