from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
history:{history}
Question: {question}

Start the answer directly. No small talk please.
"""
def set_prompt(template):
    prompt=PromptTemplate(
        template=template,
        input_variables=["context","question","history"]
    )
    return prompt


def load_model():
    llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation")
    model=ChatHuggingFace(llm=llm,)
    return model
model=load_model()

db_faiss_path="vectorstore/db_faiss"
emb_model=HuggingFaceEndpointEmbeddings(
        repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
db=FAISS.load_local(db_faiss_path,embeddings=emb_model,allow_dangerous_deserialization=True)

history=[]

rag_chain = (
    {"context": db.as_retriever(search_kwargs={"k":3}),
      "question": RunnablePassthrough(),
      "history": lambda x: "\n".join([m.content for m in history])}
    | set_prompt(template)
    | model
    | StrOutputParser()
)
while True:
    user_query=input("Write your query here: ")
    if user_query=="exit":
        print("Good bye")
        break
    history.append(HumanMessage(content=user_query))
    response=rag_chain.invoke(user_query)
    history.append(AIMessage(content=response))
    print(response)
