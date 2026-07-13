from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
import jsonlines
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pyttsx3

def send_command_to_arduino(command):
    """Send a command to Arduino."""
    try:
        arduino.write(f"{command}\n".encode('utf-8'))
        print(f"Command sent to Arduino: {command}")
    except Exception as e:
        print(f"Failed to send command to Arduino: {e}")

def text_to_speech(text):
    engine = pyttsx3.init()
    # Set the voice to Microsoft Zira
    for voice in engine.getProperty('voices'):
        if voice.id == "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_RAVI_11.0":
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

def response_rag(question, user_expression):
    file_paths = ["RAG_LLM/output-onlinetools.txt","RAG_LLM/Contact-Information2.txt","RAG_LLM/introToERC.txt"]
    docs_list = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
          docs_list.append(file.read())

    conc_docs = "\n".join(docs_list)

    embed_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url='http://127.0.0.1:11434'
    )

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=250
    )
    # Split the documents into chunks
    doc_splits = text_splitter.split_text(conc_docs)

    # Create embeddings for documents and store them in a vector store
    vectorstore = SKLearnVectorStore.from_texts(
        texts=doc_splits,
        embedding=embed_model,
    )
    retriever = vectorstore.as_retriever(k=4)

    prompt = PromptTemplate(
        template="""Your name is Vulcan.
        You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        Mood of that person is {user_expression}
        please keep the mood of the person in mind while answering the question.
        Give short answers in 2-3 sentences.
        keep the answer concise and dont say u dont know:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents", "user_expression"],
    )

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        top_p=1,
    )

    rag_chain = prompt | llm | StrOutputParser()

    # Define the RAG application class
    class RAGApplication:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain
        def run(self, question, user_expression):
            # Retrieve relevant documents
            documents = self.retriever.invoke(question)
            # Extract content from retrieved documents
            doc_texts = "\\n".join([doc.page_content for doc in documents])
            # Pass all required variables to the prompt template
            answer = self.rag_chain.invoke({
                "question": question,
                "documents": doc_texts,
                "user_expression": user_expression
            })
            return answer
        
    # Initialize the RAG application
    rag_application = RAGApplication(retriever, rag_chain)
    
    # Example usage
    answer = rag_application.run(question, user_expression)
    # print("Question:", question)
    # print("Answer:", answer)
    print("output:", answer)
    return answer
    # send_command_to_arduino(8) #mouth cycle stop 
    # text_to_speech(answer)
    # send_command_to_arduino(9) #mouth cycle stop 

# Example call
# response_rag("How are you", "neutral")
