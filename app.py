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

# Título e ícono de la página
st.set_page_config(
    page_title = "Examen DMC - LMDCBV",
    page_icon = "./favicon.ico"
)

st.header("Examen DMC ✍️")
st.subheader("Leonardo M. Del Carpio Bellido V.")
st.link_button("Fuente de conocimiento del bot", url="https://www.researchgate.net/publication/321034417_Bases_de_la_Investigacion_Cientifica")
st.divider()

# st.subheader("Fuente de conocimiento del bot: https://www.researchgate.net/publication/321034417_Bases_de_la_Investigacion_Cientifica", )

# Diseño de la barra lateral izquierda
with st.sidebar:

    st.title("Parámetros del modelo")

    model = st.selectbox('Eliga el modelo',
        (
            'gpt-3.5-turbo', 
            'gpt-3.5-turbo-16k', 
            'gpt-4'
        ), 
        key = "model"
    )

    temper = st.slider("Elije la temperatura", 0.0, 2.0, 0.5)

    max_tokens = st.slider("Elije el máximo de tokens", 5, 250, 50)

    st.text("*Mi API Key de Open AI está en los Secrets. Por favor no abusar 🙏*")

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

# Utilizando base de datos vectorial FAISS en memoria
faiss_index = FAISS.from_documents(docs, embedding_function)

# Definiendo plantilla (template) y petición (prompt)
template = """Responda a la pregunta basada en el siguiente contexto.
Si no puedes responder a la pregunta, usa la siguiente respuesta "Ignoro, papito lindo. Usa tu google nomás."

Contexto: 
{context}
Pregunta: {question}
Respuesta: 
"""

prompt = PromptTemplate(
    template = template, input_variables = ["context", "question"]
)

# Instanciando cadena de preguntas y respuestas
llm = ChatOpenAI(
    openai_api_key = api_key_openai,
    model_name = model,
    temperature = temper,
    max_tokens = max_tokens
)

# Estableciendo medio de preguntas y respuestas
qa = RetrievalQA.from_chain_type(
    llm = llm, 
    chain_type = "stuff", 
    retriever = faiss_index.as_retriever(),
    chain_type_kwargs = {"prompt": prompt, "verbose": True},
    verbose = True
)

msg_chatbot = """
        ¡Hola! Soy un chatbot integrado en OpenAI y respondo preguntas sobre la teoría del conocimiento científico:

        #### Preguntas sugeridas:
        
        - ¿Qué es el conocimiento científico?
        - ¿Cuales son los tipos de conocimiento?
        - ¿Cuáles son las características de la ciencia?
        - ¿Cuáles son los tipos de ciencia?
"""

## Se envía el prompt de usuario al modelo de GPT-3.5-Turbo para que devuelva una respuesta
def get_response_openai(prompt):
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
              
              response = get_response_openai(prompt)
              placeholder = st.empty()
              full_response = ''
              
              for item in response:
                  full_response += item
                  placeholder.markdown(full_response)

              placeholder.markdown(full_response)

      message = {"role" : "assistant", "content" : full_response}
      st.session_state.messages.append(message) #Agrega elemento a la caché de mensajes de chat.