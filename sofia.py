import speech_recognition as sr
import google.generativeai as gemini
import pyttsx3
from gtts import gTTS
import re
from dotenv import load_dotenv
import os
import json
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.io.wavfile import write, read
import pickle 


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

def extrair_mfccs(audio_data, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return mfccs.T

def treinar_modelo_gmm(dados_treino, rotulos_treino, n_components=5):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(dados_treino, rotulos_treino)
    return gmm

def ouvir_microfone():
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
        microfone.adjust_for_ambient_noise(source)
        print("Escutando...")
        audio = microfone.listen(source)
        
        try:
            # Converter para formato WAV
            audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
            sample_rate = audio.sample_rate
            write('audio.wav', sample_rate, audio_data)

            comando_usuario = microfone.recognize_google(audio, language='pt-BR')
            print("Você disse: " + comando_usuario)

            # Converter para ponto flutuante ANTES de extrair MFCCs
            audio_data = audio_data / 32768.0  # Normalizar para o intervalo [-1, 1]
            
            # Extrair MFCCs
            mfccs = extrair_mfccs(audio_data, sample_rate)

            return comando_usuario, mfccs

        except sr.UnknownValueError:
            resposta = model.generate_content("Não entendi")
            reproduzir_resposta(resposta.text)
            print("Não entendi")
            return "", None 

def reproduzir_resposta(resposta_texto):
    tts = gTTS(text=resposta_texto, lang='pt-br', tld='com')
    tts.save("resposta.mp3")
    engine.say(resposta_texto)
    engine.runAndWait()

def processar_comando(comando_usuario, falante="desconhecido"): 
    # Adaptar para usar "falante"
    if comando_usuario.lower() == "sofia parar":
        print("Encerrando o chatbot.")
        exit()
    elif comando_usuario.lower().startswith("sofia"):
        pergunta_usuario = comando_usuario[6:].strip() 
        # Usar "falante" para personalizar a resposta (se necessário)
        prompt_completo = f"{PROMPT_INICIAL}\nUsuário {falante}: {pergunta_usuario}"
        resposta = model.generate_content(prompt_completo)
        resposta_texto = remover_titulos(resposta.text)
        reproduzir_resposta(resposta_texto)
    else:
        print("Diga 'Sofia' para ativar o chatbot.")

# Carregar dados de treinamento
pasta_dados_treino = "audios"
dados_treino = []
rotulos_treino = []
for falante in os.listdir(pasta_dados_treino):
    pasta_falante = os.path.join(pasta_dados_treino, falante)
    if os.path.isdir(pasta_falante):
        for arquivo in os.listdir(pasta_falante):
            caminho_audio = os.path.join(pasta_falante, arquivo)
            
            # Ler o áudio e obter a taxa de amostragem
            sample_rate, audio_data = read(caminho_audio)

            # Converter para ponto flutuante
            audio_data = audio_data / 32768.0  # Normalizar para o intervalo [-1, 1]

            mfccs = extrair_mfccs(audio_data, sample_rate)

            dados_treino.extend(mfccs)
            rotulos_treino.extend([falante] * len(mfccs))

# Treinar o modelo GMM
gmm = treinar_modelo_gmm(dados_treino, rotulos_treino)

# Criar mapeamento de falantes (após o treinamento)
mapa_falantes = {i: falante for i, falante in enumerate(rotulos_treino)}

# Salvar o modelo treinado E o mapeamento de falantes
with open('modelo_gmm.pkl', 'wb') as arquivo_modelo:
    pickle.dump(gmm, arquivo_modelo)
with open('mapa_falantes.pkl', 'wb') as arquivo_mapa:
    pickle.dump(mapa_falantes, arquivo_mapa)

# Carregar o modelo GMM treinado E o mapeamento de falantes
with open('modelo_gmm.pkl', 'rb') as arquivo_modelo:
    gmm = pickle.load(arquivo_modelo)
with open('mapa_falantes.pkl', 'rb') as arquivo_mapa:
    mapa_falantes = pickle.load(arquivo_mapa)

while True:
    comando_usuario, mfccs = ouvir_microfone()
    if mfccs is not None:
        falante_predito = gmm.predict(mfccs)[0]
        nome_falante = mapa_falantes.get(falante_predito, "Desconhecido")  # Obter nome do falante
        print(f"Falante Predito: {nome_falante}")

        processar_comando(comando_usuario, nome_falante)
    else:
        processar_comando(comando_usuario, "desconhecido")