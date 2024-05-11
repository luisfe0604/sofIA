import speech_recognition as sr
import google.generativeai as gemini
import pyttsx3 
from gtts import gTTS
import re
from dotenv import load_dotenv
import os

load_dotenv()

gemini.configure(api_key=os.getenv("API_KEY"))
model = gemini.GenerativeModel(model_name="gemini-1.0-pro")
engine = pyttsx3.init() 
PROMPT_INICIAL = "Você é um assistente virtual amigável. Responda de forma concisa e natural, sem listas e sem palavras em negrito, como se estivesse conversando com um amigo."

def remover_titulos(texto):
  padrao = re.sub(r"^#+\s+", "", texto, flags=re.MULTILINE)
  texto_limpo = re.sub(r".*(Título|Tópico):.*\n?", "", padrao, flags=re.MULTILINE)
  return texto_limpo

def ouvir_microfone():
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
        microfone.adjust_for_ambient_noise(source)
        print("Escutando...")
        audio = microfone.listen(source)
        try:
            frase = microfone.recognize_google(audio, language='pt-BR')
            print("Você disse: " + frase)
            return frase
        except sr.UnknownValueError:
            resposta = model.generate_content("Não entendi")
            tts = gTTS(text=resposta.text, lang='pt-br', tld='com')
            tts.save("resposta.mp3")
            engine.say(resposta.text)
            engine.runAndWait()  
            print("Não entendi")
            return ""

def responder(frase):
    if frase.lower() == "sofia parar":
        print("Encerrando o chatbot.")
        exit() 
    elif frase.lower().startswith("sofia"):
        pergunta = frase[6:]

        prompt_completo = PROMPT_INICIAL + "\n" + pergunta

        resposta = model.generate_content(prompt_completo)
        resposta_texto = remover_titulos(resposta.text)
        tts = gTTS(text=resposta_texto, lang='pt-br', tld='com')
        tts.save("resposta.mp3")
        engine.say(resposta.text)
        engine.runAndWait()
    else:
        print("Diga 'Sofia' para ativar o chatbot.")

while True:
    frase = ouvir_microfone()
    responder(frase)