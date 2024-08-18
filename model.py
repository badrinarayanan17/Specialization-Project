# Suicide Prevention - Therapy ChatBot
# Guys the voice assitant feature is done, I commented out each line, understand the flow. Starting from line 89.

import os # For accessing os level dir
import pyaudio # For recording the audio
import wave # For Saving the audio results into WAVE file
import smtplib # Protocol to interact with email
from email.mime.text import MIMEText # Mime Object
from langchain_community.document_loaders import PyPDFLoader # For loading the Pdf
from langchain.prompts import PromptTemplate # Prompt template
from langchain_pinecone import PineconeVectorStore # Vector Database
from langchain.text_splitter import RecursiveCharacterTextSplitter # For Chunking
from langchain.chains import RetrievalQA # For Retrieval
from langchain_groq import ChatGroq # Inference Engine
from dotenv import load_dotenv # For detecting env variables
from langchain.embeddings import OllamaEmbeddings # To perform vector embeddings
import chainlit as cl # For user interface
from langchain_groq import ChatGroq # Inference Engine
from google.cloud import speech_v1p1beta1 as speech # Google Cloud Speech to Text API
from google.cloud.speech_v1p1beta1 import RecognitionConfig, RecognitionAudio # For audio and configurations

load_dotenv() # Detecting env 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\USER\\Downloads\\therapychatbot-432521-1f5cfe44809e.json"

# Defining prompt 
prompt_template = """ 

    You are a therapy assistant. If a person comes with sad or dull mood because of some issues, you are there to help them. Don't answer to unrelated context. If the user convey's about ending the conversation, end it. Provide response for voice input.
    Context: {context} Question: {question}

    Helpful answer:
    
"""

# Just created a function to interact with the prompt template
def set_custom_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])  
    return prompt

# Defined this function to perform retrieval
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# This function is for defining llm model
def load_llm():
    groqllm = ChatGroq(
        model ="llama3-8b-8192", temperature=0
    )
    return groqllm

# Here just loading the pdf and transforming it to chunks, and performing vector embeddings as well as storing the vector embeddings in Pinecone vector database.
def qa_bot():
    
<<<<<<< HEAD
    data = PyPDFLoader('C:\\Users\\USER\\OneDrive\\Desktop\\Specialization-Project\\TherapyBot\\Psychology-of-Human-Relations-1695056913.pdf')
=======
    data = PyPDFLoader('R:\\SPD\\TherapyBot\\Psychology-of-Human-Relations-1695056913.pdf')
>>>>>>> 45c9f06ad04fe1343085cd5cd2272424118d4820
    loader = data.load()
    chunk = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    splitdocs = chunk.split_documents(loader)
    index_name = "langchain4"
    db = PineconeVectorStore.from_documents(splitdocs[:5], OllamaEmbeddings(model ="mxbai-embed-large"),index_name=index_name)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# This functionality is for redirecting to a special email when the system found words related to suicide (static).
def send_notification(email, message):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'badrisrp3836@gmail.com'
    smtp_password = 'hngb nzfa prsd adcy'

    subject = 'Suicidal Attempt Detected'
    body = f"Conversation related to suicidal attempts:\n\n{message}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'therapybot@example.com'
    msg['To'] = email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(msg['From'], msg['To'], msg.as_string())
        
# This Functionality is to record the audio
def record_audio(filename, duration=5):
    chunk = 1024 # It defines the size of each audio chunk that is captured by the microphone. Each chunk consits of 1024 frames.
    format = pyaudio.paInt16 # 16 bit resolution for each audio sample.
    channels = 1 # Mono
    rate = 44100 # Sample rate (Number of samples of audio carried per second)

    p = pyaudio.PyAudio() # Instancing pyaudio

    # It opens the new audio stream for recording
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")

    frames = [] # Initializing the empty list 

    # This loop runs for the duration of the recording
    # No of iterations is calculated by dividing the sample rate by the chunk size and multiplying with the duration
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data) # Appending voices to the frames list

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Here we are writing the audio data to a WAV file
    # Using all the functionalities above
    wf = wave.open(filename, 'wb') 
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Conclusion - This function captures audio from the microphone, processes into chunks and saves the final result as a WAV file.
    
# Function to transcribe audio using Google Cloud Speech-to-Text
def transcribe_audio(filename):
    client = speech.SpeechClient() # Initialized the google cloud speech to text API and it will perform the transcription

    with open(filename, 'rb') as audio_file:
        content = audio_file.read() # Reading the audio file content

    audio = RecognitionAudio(content=content) 
    # It creates a Recognition Audio object, which holds the audio data that will be sent to google. The Content parameter is a binary string contaning the audio data
    
    # Configuration settings for the transcription process
    config = RecognitionConfig(
        encoding=RecognitionConfig.AudioEncoding.LINEAR16, # Standard Format
        sample_rate_hertz=44100,
        language_code='en-US'
    )

    response = client.recognize(config=config, audio=audio) # This sends the audio data and the configurations to the google cloud speech to text API. The API returns a response object, which contains the transcription.

    # Looping through the response
    for result in response.results:
        print('Transcript: {}'.format(result.alternatives[0].transcript))
        return result.alternatives[0].transcript
    
# Chainlit decorator for starting the app
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Therapy Bot. What is your query? You can also record your voice by typing 'record voice'."
    await msg.update()

    cl.user_session.set("chain", chain)
        
suicidal_keywords = ['suicide', 'self-harm', 'end my life', 'suicidal thoughts', 'kill myself']

# This is the main functionality to handle suicidal intents (Flexible to both voice and text)
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        return

    try:
        if message.content.lower() == 'record voice':
            audio_filename = 'user_audio.wav'
            record_audio(audio_filename)
            transcript = transcribe_audio(audio_filename)
            message.content = transcript

        for keyword in suicidal_keywords:
            if keyword in message.content.lower():
                email = 'badrisrp3836@gmail.com'
                send_notification(email, message.content)
                break

        res = await chain.acall({'query': message.content})
        answer = res['result']
        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()
        
        
