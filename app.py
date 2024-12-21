# Imports
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Extrayendo secrets
api_key_openai = st.secrets["API_KEY_OPENAI"]

# Cargando PDF sobre "Bases de la investigación científica"
loader = PyPDFLoader("BasesdelaInvestigacinCientfica.pdf")
data = loader.load()

# Dividiendo texto de PDF en fragmentos (Chunks)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 600, 
    chunk_overlap = 100
)
docs = text_splitter.split_documents(documents = data)

# Definiendo método de embeddings
embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

# Utilizando base de datos vectoria FAISS en memoria
faiss_index = FAISS.from_documents(docs, embedding_function)

# Definiendo template de la pregunta
template = """Responda a la pregunta basada en el siguiente contexto.
Si no puedes responder a la pregunta, usa la siguiente respuesta "No lo sé disculpa, puedes buscar la información en internet."

Contexto: 
{context}
Pregunta: {question}
Respuesta: 
"""

prompt = PromptTemplate(
    template = template, input_variables = ["context", "question"]
)

# Estableciendo modelo LLM a utilizar
llm = ChatOpenAI(
    openai_api_key = api_key_openai,
    model_name = 'gpt-3.5-turbo',
    temperature = 0.0
)

qa = RetrievalQA.from_chain_type(
    llm = llm, 
    chain_type = "stuff", 
    retriever = faiss_index.as_retriever(), #Por defecto recupera los 4 documentos más relevantes as_retriever(search_kwargs={'k': 3 })
    chain_type_kwargs = {"prompt": prompt, "verbose": True},
    verbose = True
)

st.set_page_config(
    page_title = "Examen DMC - LMDCBV",
    page_icon = "./favicon.ico"
)

with st.sidebar:

    st.title("Usando la API de OpenAI con Streamlit y Langchain")

    model = st.selectbox('Eliga el modelo',
        (
            'gpt-3.5-turbo', 
            'gpt-3.5-turbo-16k', 
            'gpt-4'
        ), 
        key = "model"
    )

    # image = Image.open('logos.png')
    # st.image(image, caption = 'OpenAI, Langchain y Streamlit')

    # st.markdown(
    #     """
    #     Integrando OpenAI con Streamlit y Langchain.
    # """
    # )

# def clear_chat_history():
#     st.session_state.messages = [{"role" : "assistant", "content": msg_chatbot}]

# st.sidebar.button('Limpiar historial de chat', on_click = clear_chat_history)



# st.write(qa.run("¿Cuales son los Tipos de conocimientos?"))

msg_chatbot = """
        Soy un chatbot que está integrado a la API de OpenAI: 

        ### Preguntas frecuentes
        
        - ¿Cuales son los Tipos de conocimientos?
"""



## Se envía el prompt de usuario al modelo de GPT-3.5-Turbo para que devuelva una respuesta
def get_response_openai(prompt, model):
    return qa.run(prompt)

#Si no existe la variable messages, se crea la variable y se muestra por defecto el mensaje de bienvenida al chatbot.
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content" : msg_chatbot}]

# Muestra todos los mensajes de la conversación
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if api_key_openai:

  prompt = st.chat_input("Ingresa tu pregunta")
  if prompt:
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.write(prompt)

  # Generar una nueva respuesta si el último mensaje no es de un assistant, sino de un user, entonces entra al bloque de código
  if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
          with st.spinner("Esperando respuesta, dame unos segundos."):
              
              response = get_response_openai(prompt, model)
              placeholder = st.empty()
              full_response = ''
              
              for item in response:
                  full_response += item
                  placeholder.markdown(full_response)

              placeholder.markdown(full_response)

      message = {"role" : "assistant", "content" : full_response}
      st.session_state.messages.append(message) #Agrega elemento a la caché de mensajes de chat.