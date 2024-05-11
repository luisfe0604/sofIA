import speech_recognition as sr
import google.generativeai as gemini
import pyttsx3
from gtts import gTTS
import re
from dotenv import load_dotenv
import os
import json

load_dotenv()
gemini.configure(api_key=os.getenv("API_KEY"))
model = gemini.GenerativeModel(model_name="gemini-1.0-pro")
engine = pyttsx3.init()

PROMPT_INICIAL = "Você é um assistente virtual amigável. Responda de forma concisa e natural, sem listas e sem palavras em negrito, como se estivesse conversando com um amigo."

with open('respostas.json', 'r') as f:
    respostas_predefinidas = json.load(f)

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
            comando_usuario = microfone.recognize_google(audio, language='pt-BR')
            print("Você disse: " + comando_usuario)
            return comando_usuario
        except sr.UnknownValueError:
            resposta = model.generate_content("Não entendi")
            reproduzir_resposta(resposta.text)
            print("Não entendi")
            return ""

def reproduzir_resposta(resposta_texto):
    tts = gTTS(text=resposta_texto, lang='pt-br', tld='com')
    tts.save("resposta.mp3")
    engine.say(resposta_texto)
    engine.runAndWait()

def processar_comando(comando_usuario):
    if comando_usuario.lower() == "sofia parar":
        print("Encerrando o chatbot.")
        exit()
    elif comando_usuario.lower().startswith("sofia"):
        pergunta_usuario = comando_usuario[6:].strip()        
        resposta_predefinida = respostas_predefinidas.get(pergunta_usuario.lower())
        if resposta_predefinida:
            reproduzir_resposta(resposta_predefinida)
        else:
            prompt_completo = PROMPT_INICIAL + "\n" + pergunta_usuario
            resposta = model.generate_content(prompt_completo)
            resposta_texto = remover_titulos(resposta.text)
            reproduzir_resposta(resposta_texto)
    else:
        print("Diga 'Sofia' para ativar o chatbot.")

while True:
    comando_usuario = ouvir_microfone()
    processar_comando(comando_usuario)