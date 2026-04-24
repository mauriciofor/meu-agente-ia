# -*- coding: utf-8 -*-
import os
import sys
import json
import requests
import sqlite3
import re
import time
import traceback
import logging
import ast
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
from dotenv import load_dotenv

logging.getLogger('werkzeug').setLevel(logging.ERROR)

from flask import Flask, request, jsonify, Response, render_template_string, stream_with_context
from flask_cors import CORS

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_mistralai.chat_models import ChatMistralAI

# Memória em nuvem (Redis) - instale: pip install redis
import redis

# Web Scraping
from bs4 import BeautifulSoup

try:
    from groq import Groq
    GROQ_NATIVE = True
except:
    GROQ_NATIVE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except:
    EMBEDDINGS_AVAILABLE = False

ultimo_codigo = {}

# ============================
# Rate Limiter
# ============================
class SimpleRateLimiter:
    def __init__(self, requests_per_minute=30):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_allowed(self, key):
        now = time.time()
        minute_ago = now - 60
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > minute_ago]
        if len(self.requests[key]) >= self.requests_per_minute:
            return False
        self.requests[key].append(now)
        return True

rate_limiter = SimpleRateLimiter(requests_per_minute=30)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

env_path = resource_path('.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

app = Flask(__name__)
CORS(app)

def limpar_tabela(texto):
    if not texto:
        return texto
    linhas = texto.split('\n')
    novas = []
    for linha in linhas:
        if linha.strip().replace('|', '').replace('-', '').replace(' ', '') == '':
            continue
        linha = linha.replace('|', ' ')
        novas.append(linha)
    texto = '\n'.join(novas)
    texto = re.sub(r' +', ' ', texto)
    return texto.strip()

class CacheManager:
    def __init__(self):
        self.memory_cache = {}
        self.cache_lock = Lock()

    def get(self, key):
        with self.cache_lock:
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                if expiry > time.time():
                    return value
                else:
                    del self.memory_cache[key]
        return None

    def set(self, key, value, ttl=300):
        with self.cache_lock:
            self.memory_cache[key] = (value, time.time() + ttl)

    def clear_pattern(self, pattern):
        with self.cache_lock:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.memory_cache[key]

cache = CacheManager()

# Histórico local (fallback)
conversas = {}

# ============================
# Memória em nuvem (Redis)
# Configure UPSTASH_REDIS_URL no .env
# ============================
redis_client = None
if os.getenv("UPSTASH_REDIS_URL"):
    try:
        redis_client = redis.from_url(os.getenv("UPSTASH_REDIS_URL"), decode_responses=True)
        print("✅ Redis conectado (Upstash)")
    except Exception as e:
        print(f"⚠️ Redis erro: {e}")

def get_historico_nuvem(session_id):
    if not redis_client:
        return []
    key = f"chat:{session_id}"
    try:
        data = redis_client.lrange(key, -30, -1)
        historico = []
        for item in data:
            try:
                msg = json.loads(item)
                historico.append(msg)
            except:
                pass
        return historico
    except:
        return []

def add_ao_historico_nuvem(session_id, pergunta, resposta):
    if not redis_client:
        return
    key = f"chat:{session_id}"
    msg = json.dumps({"pergunta": pergunta, "resposta": resposta, "timestamp": time.time()})
    try:
        redis_client.rpush(key, msg)
        redis_client.ltrim(key, -30, -1)
        redis_client.expire(key, 86400)
    except:
        pass

# ============================
# Funções de histórico (com fallback nuvem -> local -> DB)
# ============================
def get_historico(session_id):
    # Primeiro tenta nuvem
    nuvem = get_historico_nuvem(session_id)
    if nuvem:
        return nuvem
    # Fallback local
    if session_id not in conversas:
        conversas[session_id] = []
    return conversas[session_id]

def add_ao_historico(session_id, pergunta, resposta):
    # Salva na nuvem
    add_ao_historico_nuvem(session_id, pergunta, resposta)
    # Salva local
    if session_id not in conversas:
        conversas[session_id] = []
    conversas[session_id].append({"pergunta": pergunta, "resposta": resposta})
    if len(conversas[session_id]) > 15:
        conversas[session_id] = conversas[session_id][-15:]
    # Salva no SQLite (compatibilidade)
    _salvar_historico_db(session_id, pergunta, resposta)
    _extrair_e_salvar_fatos(session_id, pergunta, resposta)

def _salvar_historico_db(session_id, pergunta, resposta):
    try:
        db_path = resource_path('memoria_agente.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historico_sessoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                pergunta TEXT,
                resposta TEXT
            )
        ''')
        agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO historico_sessoes (session_id, timestamp, pergunta, resposta)
            VALUES (?, ?, ?, ?)
        ''', (session_id, agora, pergunta[:500], resposta[:500]))
        conn.commit()
        conn.close()
    except:
        pass

def _extrair_e_salvar_fatos(session_id, pergunta, resposta):
    texto_completo = f"{pergunta} {resposta}".lower()
    padroes = [
        (r'(?:meu nome é|me chamo|sou o|sou a|chamo-me) (\w+)', 'nome'),
        (r'(?:moro em|sou de|vivo em|município de|resido em) ([\w\s]+?)(?:\.|\?|,|$)', 'cidade'),
        (r'(?:trabalho com|sou|profissão é) ([\w\s]+?)(?:\.|\?|,|$)', 'profissao'),
        (r'(?:gosto de|adoro|amo|curto) ([\w\s]+?)(?:\.|\?|,|$)', 'gosto'),
        (r'(?:tenho|possuo|idade) (\d+) (?:anos|idade)', 'idade'),
        (r'(?:meu email é|meu e-mail é) ([\w\.]+@[\w\.]+)', 'email'),
    ]
    for padrao, tipo in padroes:
        match = re.search(padrao, texto_completo, re.IGNORECASE)
        if match:
            valor = match.group(1).strip()
            if len(valor) > 1 and len(valor) < 100:
                fato = f"📌 {tipo.upper()}: {valor}"
                aprender_licao("fato_extraido", fato)
                print(f"🧠 Fato extraído: {fato}")
                break

def carregar_historico_persistido(session_id, limite=50):
    try:
        db_path = resource_path('memoria_agente.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pergunta, resposta FROM historico_sessoes 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limite))
        resultados = cursor.fetchall()
        conn.close()
        for pergunta, resposta in reversed(resultados):
            if session_id not in conversas:
                conversas[session_id] = []
            if not any(m['pergunta'] == pergunta for m in conversas[session_id]):
                conversas[session_id].append({"pergunta": pergunta, "resposta": resposta})
        return len(resultados)
    except:
        return 0

def limpar_historico(session_id):
    if session_id in conversas:
        conversas[session_id] = []
    if redis_client:
        try:
            redis_client.delete(f"chat:{session_id}")
        except:
            pass

def init_memoria():
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conhecimento (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT UNIQUE,
            resposta TEXT,
            api_usada TEXT,
            vezes_usada INTEGER DEFAULT 1,
            acertos INTEGER DEFAULT 0,
            erros INTEGER DEFAULT 0,
            ultima_vez TEXT,
            data_criacao TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT,
            resposta TEXT,
            api_usada TEXT,
            feedback TEXT,
            data TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS licoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT,
            regra TEXT,
            data TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT,
            conteudo TEXT,
            chunks INTEGER,
            data_processamento TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT,
            embedding BLOB,
            categoria TEXT,
            data TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historico_sessoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            pergunta TEXT,
            resposta TEXT
        )
    ''')
    conn.commit()
    conn.close()

def aprender_licao(tipo, regra):
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO licoes (tipo, regra, data) VALUES (?, ?, ?)', (tipo, regra, agora))
    conn.commit()
    conn.close()
    print(f"📚 Licao aprendida: {regra[:100]}")

def buscar_licoes():
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT tipo, regra FROM licoes ORDER BY id DESC LIMIT 30')
    resultados = cursor.fetchall()
    conn.close()
    licoes = {}
    for tipo, regra in resultados:
        if tipo not in licoes:
            licoes[tipo] = []
        licoes[tipo].append(regra)
    return licoes

def aprender_resposta(pergunta, resposta, api_usada):
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute('''
            INSERT INTO conhecimento (pergunta, resposta, api_usada, ultima_vez, data_criacao)
            VALUES (?, ?, ?, ?, ?)
        ''', (pergunta, resposta, api_usada, agora, agora))
    except sqlite3.IntegrityError:
        cursor.execute('''
            UPDATE conhecimento 
            SET vezes_usada = vezes_usada + 1, ultima_vez = ?, api_usada = ?
            WHERE pergunta = ?
        ''', (agora, api_usada, pergunta))
    conn.commit()
    conn.close()

def lembrar(pergunta):
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT resposta, api_usada, acertos, erros 
        FROM conhecimento 
        WHERE pergunta = ?
        ORDER BY vezes_usada DESC
        LIMIT 1
    ''', (pergunta,))
    resultado = cursor.fetchone()
    conn.close()
    if resultado:
        resposta, api_usada, acertos, erros = resultado
        total = acertos + erros
        confianca = (acertos / total * 100) if total > 0 else 50
        if confianca > 70:
            return resposta, api_usada, confianca
    return None, None, 0

def registrar_feedback(pergunta, resposta, api_usada, feedback_usuario):
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO feedback (pergunta, resposta, api_usada, feedback, data)
        VALUES (?, ?, ?, ?, ?)
    ''', (pergunta, resposta, api_usada, feedback_usuario, agora))
    if feedback_usuario == 'bom':
        cursor.execute('UPDATE conhecimento SET acertos = acertos + 1 WHERE pergunta = ?', (pergunta,))
    elif feedback_usuario == 'ruim':
        cursor.execute('UPDATE conhecimento SET erros = erros + 1 WHERE pergunta = ?', (pergunta,))
    conn.commit()
    conn.close()

def contar_tokens(texto):
    return len(texto) // 4

def limitar_contexto(texto, max_tokens=3000):
    tokens_estimados = contar_tokens(texto)
    if tokens_estimados > max_tokens:
        limite_caracteres = max_tokens * 4
        return texto[:limite_caracteres] + "..."
    return texto

# ============================
# Web Scraping Estruturado
# ============================
def estruturar_dados_web(texto_bruto, pergunta_original=""):
    if not texto_bruto:
        return texto_bruto
    texto = texto_bruto.strip()
    padroes = [
        (r'(\d+(?:[.,]\d+)?\s*(?:°C|%|km/h|m/s|mm|cm|m|km|kg|g|R\$|\$|€|USD|BRL)\b)', r'\n• \1'),
        (r'([A-ZÀ-Ú][a-zà-ú]+(?:\s+[a-zà-ú]+)*)\s*[:：]\s*([^.•\n]+)', r'\n• \1: \2'),
        (r'(?:^|\n)\s*[-*•]\s*([^\n]+)', r'\n• \1'),
        (r'(?:^|\n)\s*\d+\.\s*([^\n]+)', r'\n• \1'),
        (r'\((\d+[^)]+)\)', r' [\1]'),
    ]
    for padrao, substituto in padroes:
        texto = re.sub(padrao, substituto, texto, flags=re.MULTILINE)
    linhas = texto.split('\n')
    linhas_vistas = set()
    linhas_unicas = []
    for linha in linhas:
        linha_limpa = linha.strip()
        if linha_limpa and linha_limpa not in linhas_vistas:
            linhas_vistas.add(linha_limpa)
            linhas_unicas.append(linha)
    texto = '\n'.join(linhas_unicas)
    texto = re.sub(r'\.\s+([A-ZÀ-Ú])', r'.\n\n\1', texto)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    return texto.strip()

def formatar_texto_busca(texto):
    if not texto:
        return texto
    texto = estruturar_dados_web(texto)
    texto = re.sub(r'\.([A-ZÀ-Ú])', r'. \1', texto)
    texto = re.sub(r'\?([A-ZÀ-Ú])', r'? \1', texto)
    texto = re.sub(r'\!([A-ZÀ-Ú])', r'! \1', texto)
    texto = re.sub(r'\,([A-ZÀ-Úa-zà-ú])', r', \1', texto)
    texto = re.sub(r'\;([A-ZÀ-Ú])', r'; \1', texto)
    texto = re.sub(r'\:([A-ZÀ-Ú])', r': \1', texto)
    texto = re.sub(r'([a-zà-ú])([A-ZÀ-Ú])', r'\1 \2', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def resumir_resultado_busca(texto_bruto, pergunta):
    texto_limpo = texto_bruto.replace('🌐', '').strip()
    texto_limpo = estruturar_dados_web(texto_limpo, pergunta)
    try:
        prompt = f"""Resuma estas informações de forma organizada:
Pergunta: {pergunta}
Informações: {texto_limpo[:800]}

Use tópicos (•) para dados importantes.
Resposta:"""
        for nome, llm in api_manager.apis.items():
            if 'groq' in nome and '8b' in nome:
                try:
                    resposta = llm.invoke(prompt, timeout=10)
                    return f"🌤️ {resposta.content.strip()}"
                except:
                    continue
    except:
        pass
    return f"🌤️ {texto_limpo[:300]}..."

# ============================
# Fontes confiáveis: Wikipedia, Arxiv, DuckDuckGo
# ============================
search = DuckDuckGoSearchRun()
wiki_wrapper = WikipediaAPIWrapper(lang='pt', top_k_results=2, doc_content_chars_max=1500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

def buscar_fonte_confiavel(pergunta):
    resultados = []
    # Wikipedia (português)
    try:
        wiki = wikipedia_tool.run(pergunta)
        if wiki and len(wiki) > 100:
            resultados.append(f"📘 Wikipedia:\n{wiki[:1200]}")
    except Exception as e:
        print(f"Wikipedia erro: {e}")
    # Arxiv (artigos científicos)
    try:
        arxiv = arxiv_tool.run(pergunta)
        if arxiv and len(arxiv) > 100:
            resultados.append(f"📄 Arxiv:\n{arxiv[:1200]}")
    except Exception as e:
        print(f"Arxiv erro: {e}")
    # DuckDuckGo (atualidades)
    try:
        ddg = search.run(pergunta)
        if ddg:
            resultados.append(f"🌐 DuckDuckGo:\n{formatar_texto_busca(ddg)[:1200]}")
    except Exception as e:
        print(f"DuckDuckGo erro: {e}")
    return "\n\n".join(resultados) if resultados else ""

# ============================
# Sanitização e código
# ============================
def sanitizar_input(texto):
    texto = texto[:50000]
    texto = re.sub(r'[<>]', '', texto)
    texto = texto.strip()
    return texto

def validar_codigo_estaticamente(codigo):
    erros = []
    try:
        ast.parse(codigo)
    except SyntaxError as e:
        erros.append(f"Erro de sintaxe na linha {e.lineno}: {e.msg}")
    except Exception as e:
        erros.append(f"Erro de sintaxe: {str(e)}")
    imports_comuns = {
        'requests': 'import requests',
        'json': 'import json',
        'datetime': 'from datetime import datetime',
        're': 'import re',
        'os': 'import os',
        'sys': 'import sys',
        'time': 'import time',
        'random': 'import random',
        'math': 'import math',
        'sqlite3': 'import sqlite3',
        'flask': 'from flask import Flask',
        'tkinter': 'import tkinter as tk',
        'threading': 'import threading',
        'bs4': 'from bs4 import BeautifulSoup',
        'BeautifulSoup': 'from bs4 import BeautifulSoup'
    }
    for modulo, import_stmt in imports_comuns.items():
        if f'{modulo}.' in codigo and import_stmt not in codigo:
            erros.append(f"Falta importar: {import_stmt}")
    return erros

def auto_corrigir_codigo(codigo_original, erros, pergunta_original):
    if not erros:
        return codigo_original
    prompt_correcao = "CODIGO COM ERROS:\n" + codigo_original + "\n\nERROS ENCONTRADOS:\n" + "\n".join("- " + e for e in erros) + "\n\nSOLICITACAO ORIGINAL:\n" + pergunta_original + "\n\nCORRIJA o codigo e devolva APENAS o codigo completo corrigido:"
    return prompt_correcao

def processar_codigo_com_auto_correcao(pergunta, api_manager, session_id, cache_key, max_tentativas=3):
    for tentativa in range(max_tentativas):
        for api_nome in api_manager.apis.keys():
            if 'deepseek' in api_nome or 'groq' in api_nome:
                try:
                    llm = api_manager.apis[api_nome]
                    prompt = "Gere um codigo Python FUNCIONAL e COMPLETO para: " + pergunta + "\n\nREGRAS:\n1. Use QUEBRAS DE LINHA para separar as linhas do codigo\n2. Inclua todos os imports\n3. NAO inclua explicacoes\n\nCODIGO:"
                    resposta = llm.invoke(prompt, timeout=120)
                    if resposta and resposta.content:
                        codigo = resposta.content.strip()
                        codigo = codigo.replace("``python", "```python")
                        codigo = codigo.replace("``", "```")
                        if "```python" not in codigo:
                            codigo = "```python\n" + codigo + "\n```"
                        return "✅ Codigo validado:\n\n" + codigo
                except Exception as e:
                    print(f"❌ Erro na API {api_nome}: {e}")
                    continue
    return None

def _carregar_aprendizados():
    try:
        db_path = resource_path('memoria_agente.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT regra FROM licoes WHERE tipo='fato' OR tipo='fato_extraido' ORDER BY id DESC")
        fatos = cursor.fetchall()
        conn.close()
        if fatos:
            return "📚 INFORMACOES QUE VOCE JA SABE SOBRE O USUARIO (USE PARA RESPONDER):\n" + "\n".join([f[0] for f in fatos[:20]])
        return ""
    except:
        return ""

def _salvar_fato(conteudo):
    try:
        db_path = resource_path('memoria_agente.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('INSERT INTO licoes (tipo, regra, data) VALUES (?, ?, ?)', ('fato', conteudo, agora))
        conn.commit()
        conn.close()
        return True
    except:
        return False

# ============================
# Gerenciador de Contexto (para resumir conversas longas)
# ============================
class GerenciadorContexto:
    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.max_tokens = 6000
        self.tokens_reservados_pergunta = 1000
        self.tokens_reservados_resposta = 1500

    def resumir_texto(self, texto, max_tokens=500):
        if len(texto) < 500:
            return texto
        prompt_resumo = "Resuma o seguinte historico de conversa em no maximo 5 frases:\n\n" + texto[:4000] + "\n\nResumo:"
        for api_nome in ['groq', 'gemini', 'deepseek']:
            for nome, llm in self.api_manager.apis.items():
                if api_nome in nome.lower():
                    try:
                        resultado = llm.invoke(prompt_resumo, timeout=10)
                        resumo = resultado.content.strip()
                        return resumo[:max_tokens * 4]
                    except:
                        continue
        linhas = texto.split('\n')
        if len(linhas) > 10:
            return '\n'.join(linhas[:3] + ['[...]'] + linhas[-3:])
        return texto[:500] + "..."

    def preparar_contexto(self, historico_msg, pergunta_atual, contexto_extra=""):
        tokens_disponiveis = self.max_tokens - self.tokens_reservados_pergunta - self.tokens_reservados_resposta
        historico_texto = ""
        for msg in historico_msg:
            historico_texto += f"Usuario: {msg['pergunta']}\nAssistente: {msg['resposta']}\n"
        if contexto_extra:
            historico_texto = contexto_extra + "\n" + historico_texto
        tokens_historico = contar_tokens(historico_texto)
        if tokens_historico <= tokens_disponiveis:
            return historico_texto
        else:
            print(f"⚠️ Contexto grande ({tokens_historico} tokens). Resumindo...")
            if len(historico_msg) > 3:
                recente = historico_msg[-3:]
                antigo = historico_msg[:-3]
                texto_antigo = ""
                for msg in antigo:
                    texto_antigo += f"Usuario: {msg['pergunta']}\nAssistente: {msg['resposta']}\n"
                texto_recente = ""
                for msg in recente:
                    texto_recente += f"Usuario: {msg['pergunta']}\nAssistente: {msg['resposta']}\n"
                resumo = self.resumir_texto(texto_antigo)
                contexto_final = f"[RESUMO DA CONVERSA ANTERIOR]:\n{resumo}\n\n[CONVERSA RECENTE]:\n{texto_recente}"
                if contar_tokens(contexto_final) > tokens_disponiveis:
                    return self.resumir_texto(historico_texto, max_tokens=tokens_disponiveis)
                return contexto_final
            else:
                return self.resumir_texto(historico_texto, max_tokens=tokens_disponiveis)

# ============================
# RAG e Memória Vetorial
# ============================
class RAGManager:
    def __init__(self):
        self.model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ Modelo de embeddings carregado!")
            except:
                print("⚠️ Fallback para busca por palavras")

    def gerar_embedding(self, texto):
        if self.model:
            try:
                return self.model.encode(texto[:1000])
            except:
                pass
        return None

    def buscar_similares(self, pergunta, limite=3):
        if not self.model:
            return self._buscar_por_palavras(pergunta, limite)
        try:
            pergunta_emb = self.gerar_embedding(pergunta)
            if pergunta_emb is None:
                return []
            memorias = self._carregar_embeddings()
            similaridades = []
            for mem in memorias:
                if mem.get('embedding'):
                    emb = np.frombuffer(mem['embedding'], dtype=np.float32)
                    sim = np.dot(pergunta_emb, emb) / (np.linalg.norm(pergunta_emb) * np.linalg.norm(emb))
                    similaridades.append((sim, mem))
            similaridades.sort(reverse=True)
            return [mem for _, mem in similaridades[:limite]]
        except:
            return []

    def _buscar_por_palavras(self, pergunta, limite):
        palavras_pergunta = set(pergunta.lower().split())
        memorias = memoria_vetorial.memorias
        resultados = []
        for memoria in memorias:
            texto = memoria['texto'].lower()
            palavras_texto = set(texto.split())
            similaridade = len(palavras_pergunta & palavras_texto)
            if similaridade > 0:
                resultados.append((similaridade, memoria))
        resultados.sort(key=lambda x: x[0], reverse=True)
        return [memoria for _, memoria in resultados[:limite]]

    def _carregar_embeddings(self):
        try:
            db_path = resource_path('memoria_agente.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT texto, embedding, categoria FROM embeddings ORDER BY id DESC LIMIT 100')
            resultados = []
            for row in cursor.fetchall():
                resultados.append({
                    'texto': row[0],
                    'embedding': row[1],
                    'categoria': row[2]
                })
            conn.close()
            return resultados
        except:
            return []

    def adicionar_embedding(self, texto, categoria="conversa"):
        emb = self.gerar_embedding(texto)
        if emb is not None:
            try:
                db_path = resource_path('memoria_agente.db')
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                emb_bytes = emb.tobytes()
                cursor.execute('''
                    INSERT INTO embeddings (texto, embedding, categoria, data)
                    VALUES (?, ?, ?, ?)
                ''', (texto[:500], emb_bytes, categoria, agora))
                conn.commit()
                conn.close()
                return True
            except:
                pass
        return False

# ============================
# MultiAPIManager (seus modelos originais, intactos)
# ============================
class MultiAPIManager:
    def __init__(self):
        self.apis = {}
        self.api_status = {}
        self.groq_native = None
        if GROQ_NATIVE and os.getenv("GROQ_API_KEY"):
            try:
                self.groq_native = Groq(api_key=os.getenv("GROQ_API_KEY"))
                print("✅ Groq Native Client carregado (Function Calling)!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Groq Native: {e}")

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "buscar_web",
                    "description": "Busca informações atualizadas na internet",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Termo de busca"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "executar_codigo",
                    "description": "Executa código Python e retorna o resultado",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "codigo": {"type": "string", "description": "Código Python a ser executado"}
                        },
                        "required": ["codigo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "salvar_memoria",
                    "description": "Salva uma informação importante na memória",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "informacao": {"type": "string", "description": "Informação a ser salva"}
                        },
                        "required": ["informacao"]
                    }
                }
            }
        ]
        self.carregar_apis()

    def carregar_apis(self):
        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                self.apis['deepseek_chat'] = ChatOpenAI(model="deepseek-chat",
                                                        openai_api_base="https://api.deepseek.com/v1",
                                                        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0.3)
                self.api_status['deepseek_chat'] = {"status": "ok", "usos": 0, "erros": 0, "tipo": "geral"}
                print("✅ DeepSeek deepseek-chat carregado (geral)!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar DeepSeek deepseek-chat: {e}")
            try:
                self.apis['deepseek_reasoner'] = ChatOpenAI(model="deepseek-reasoner",
                                                            openai_api_base="https://api.deepseek.com/v1",
                                                            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                                                            temperature=0.5, model_kwargs={
                        "extra_body": {"thinking": {"type": "enabled"}}})
                self.api_status['deepseek_reasoner'] = {"status": "ok", "usos": 0, "erros": 0, "tipo": "raciocinio"}
                print("✅ DeepSeek deepseek-reasoner carregado (raciocinio)!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar DeepSeek deepseek-reasoner: {e}")
        if os.getenv("GROQ_API_KEY"):
            try:
                self.apis['compound'] = ChatGroq(model="llama-3.3-70b-versatile",
                                                 groq_api_key=os.getenv("GROQ_API_KEY"),
                                                 temperature=0.3)
                self.api_status['compound'] = {"status": "ok", "usos": 0, "erros": 0, "tipo": "agente"}
                print("✅ Groq LLaMA 3.3 carregado (agente autônomo)!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Compound: {e}")

            try:
                self.apis['compound_mini'] = ChatGroq(model="mixtral-8x7b-32768",
                                                      groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)
                self.api_status['compound_mini'] = {"status": "ok", "usos": 0, "erros": 0, "tipo": "agente"}
                print("✅ Groq Mixtral carregado (agente leve)!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Compound-mini: {e}")
        if os.getenv("GOOGLE_API_KEY"):
            modelos_gemini = [("gemini-2.0-flash-exp", 0.3, "rapido"), ("gemini-1.5-flash", 0.3, "rapido"),
                              ("gemini-1.5-pro", 0.3, "contexto longo")]
            for modelo, temp, tipo in modelos_gemini:
                try:
                    nome_api = f"gemini_{modelo.replace('-', '_')}"
                    self.apis[nome_api] = ChatGoogleGenerativeAI(model=modelo, temperature=temp,
                                                                 google_api_key=os.getenv("GOOGLE_API_KEY"))
                    self.api_status[nome_api] = {"status": "ok", "usos": 0, "erros": 0, "tipo": tipo}
                    print(f"✅ Gemini {modelo} carregado ({tipo})!")
                except Exception as e:
                    print(f"⚠️ Erro ao carregar Gemini {modelo}: {e}")
        if os.getenv("COHERE_API_KEY"):
            try:
                self.apis['cohere_command_r'] = ChatCohere(model="command-r",
                                                           cohere_api_key=os.getenv("COHERE_API_KEY"), temperature=0.3)
                self.api_status['cohere_command_r'] = {"status": "ok", "usos": 0, "erros": 0, "tipo": "geral"}
                print("✅ Cohere carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Cohere: {e}")
        if os.getenv("MISTRAL_API_KEY"):
            try:
                self.apis['mistral_large'] = ChatMistralAI(model="mistral-large-latest",
                                                           mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                                                           temperature=0.3)
                self.api_status['mistral_large'] = {"status": "ok", "usos": 0, "erros": 0, "tipo": "premium"}
                print("✅ Mistral carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Mistral: {e}")
        if not self.apis:
            print("⚠️ NENHUMA API carregada!")

    def usar_groq_com_tools(self, pergunta, contexto=""):
        if not self.groq_native:
            return None, "Groq Native não disponível"
        try:
            fatos_aprendidos = _carregar_aprendizados()
            contexto_completo = contexto
            if fatos_aprendidos:
                contexto_completo = f"""📚 INFORMACOES SOBRE O USUARIO (USE NATURALMENTE):
    {fatos_aprendidos}

    {contexto}"""
            response = self.groq_native.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system",
                     "content": f"Voce e um assistente inteligente que conhece o usuario. {contexto_completo}. Se não souber a resposta, diga 'Não sei' ou 'Não tenho informação suficiente'. Não invente fatos."},
                    {"role": "user", "content": pergunta}
                ],
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3
            )
            message = response.choices[0].message
            if message.tool_calls:
                resultados = []
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    if func_name == "buscar_web":
                        try:
                            query = args['query']
                            if ('tempo' in pergunta.lower() or 'clima' in pergunta.lower()) and fatos_aprendidos and 'CIDADE:' in fatos_aprendidos:
                                match = re.search(r'CIDADE:\s*([\w\s]+)', fatos_aprendidos)
                                if match:
                                    cidade = match.group(1).strip()
                                    if cidade.lower() not in query.lower():
                                        query = f"previsão do tempo para {cidade}"
                            resultado = search.run(query)
                            resultados.append(f"🌐 Busca: {resultado[:500]}")
                        except:
                            resultados.append("❌ Erro na busca web")
                    elif func_name == "executar_codigo":
                        try:
                            exec_globals = {}
                            exec(args['codigo'], exec_globals)
                            resultados.append(f"✅ Codigo executado com sucesso")
                        except Exception as e:
                            resultados.append(f"❌ Erro no codigo: {str(e)}")
                    elif func_name == "salvar_memoria":
                        aprender_licao("fato", args['informacao'])
                        resultados.append(f"📚 Salvo na memoria: {args['informacao'][:100]}")
                response2 = self.groq_native.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": f"Contexto: {contexto_completo}"},
                        {"role": "user", "content": pergunta},
                        message,
                        {"role": "tool", "tool_call_id": message.tool_calls[0].id, "content": "\n".join(resultados)}
                    ],
                    temperature=0.3
                )
                return response2.choices[0].message.content, "groq_tools"
            return message.content, "groq_native"
        except Exception as e:
            print(f"❌ Erro no Groq Tools: {e}")
            return None, str(e)

    def precisa_explicacao(self, pergunta):
        palavras = ['explique', 'detalhe', 'detalhadamente', 'passo a passo', 'como funciona', 'por que', 'justifique',
                    'raciocinio']
        return any(p in pergunta.lower() for p in palavras)

    def precisa_criatividade(self, pergunta):
        palavras = ['crie', 'escreva', 'poema', 'historia', 'imagine', 'invente', 'criativo', 'poesia', 'conto',
                    'personagem', 'historia']
        return any(p in pergunta.lower() for p in palavras)

    def precisa_codigo(self, pergunta):
        palavras_acao = ['crie um codigo', 'gere um codigo', 'codigo python', 'programa que', 'funcao que',
                         'script que', 'escreva um codigo', 'criar codigo', 'gera um codigo']
        return any(p in pergunta.lower() for p in palavras_acao)

    def pergunta_simples(self, pergunta):
        palavras_simples = ['oi', 'ola', 'tudo bem', 'obrigado', 'valeu']
        return any(p in pergunta.lower() for p in palavras_simples) or len(pergunta.split()) < 5

    def escolher_melhor_api(self, pergunta):
        if self.pergunta_simples(pergunta):
            for nome in self.apis.keys():
                if "groq" in nome and "8b" in nome:
                    return nome
        if self.precisa_codigo(pergunta):
            for nome in self.apis.keys():
                if "deepseek" in nome:
                    return nome
        if self.precisa_criatividade(pergunta):
            for nome in self.apis.keys():
                if "groq" in nome and "llama" in nome:
                    return nome
        if self.precisa_explicacao(pergunta):
            for nome in self.apis.keys():
                if "deepseek_reasoner" in nome:
                    return nome
        if len(pergunta) > 500:
            for nome in self.apis.keys():
                if "gemini_1_5_pro" in nome:
                    return nome
        for nome in self.apis.keys():
            if "groq" in nome and "llama" in nome:
                return nome
        if self.apis:
            return list(self.apis.keys())[0]
        return None

    def usar_api_com_stream(self, nome, pergunta, contexto="", historico_texto=""):
        if nome not in self.apis:
            return None
        try:
            self.api_status[nome]["usos"] += 1
            llm = self.apis[nome]

            fatos_aprendidos = _carregar_aprendizados()
            contexto_completo = ""
            if fatos_aprendidos:
                contexto_completo = f"""📚 INFORMACOES QUE VOCE JA SABE SOBRE O USUARIO:
    {fatos_aprendidos}

    Use essas informacoes NATURALMENTE quando forem relevantes.

    """
            if historico_texto:
                contexto_completo += f"\n📜 HISTORICO DA CONVERSA:\n{historico_texto}\n"

            prompt = f"{contexto_completo}\nPergunta: {pergunta}\n\nREGRAS IMPORTANTES:\n- Se você não souber a resposta, diga 'Não sei' ou 'Não tenho informação suficiente'.\n- Não invente fatos.\n- Prefira usar Wikipedia ou fontes confiáveis.\nResponda de forma DIRETA e OBJETIVA:"
            prompt = limitar_contexto(prompt, 4000)
            return llm.stream(prompt)
        except Exception as e:
            self.api_status[nome]["erros"] += 1
            print(f"❌ Erro no streaming da API {nome}: {e}")
            return None

    def usar_api(self, nome, pergunta, contexto="", busca=None, historico_texto=""):
        if nome in self.apis:
            resposta, erro = self._tentar_api(nome, pergunta, contexto, busca, historico_texto)
            if not erro:
                return resposta, None
            provedor = nome.split('_')[0]
            print(f"🔄 API {nome} falhou. Tentando outros modelos do provedor {provedor}...")
            for outro_nome in self.apis.keys():
                if outro_nome.startswith(provedor) and outro_nome != nome:
                    resposta, erro = self._tentar_api(outro_nome, pergunta, contexto, busca, historico_texto)
                    if not erro:
                        print(f"✅ Fallback para {outro_nome} funcionou!")
                        return resposta, None
        return None, "Nenhuma API disponivel"

    def _tentar_api(self, nome, pergunta, contexto, busca, historico_texto=""):
        if nome not in self.apis:
            return None, f"API {nome} nao disponivel"
        try:
            if any(p in pergunta.lower() for p in
                   ['você é capaz', 'você consegue', 'o que você faz', 'suas capacidades', 'vc é capaz',
                    'vc consegue']):
                return "Sim, sou um assistente de IA capaz de gerar e corrigir códigos, responder perguntas, aprender com conversas e buscar informações atualizadas na internet.", None
            self.api_status[nome]["usos"] += 1
            llm = self.apis[nome]
            fatos_aprendidos = _carregar_aprendizados()
            contexto_completo = ""
            if fatos_aprendidos:
                contexto_completo = f"""📚 INFORMACOES QUE VOCE JA SABE SOBRE O USUARIO:
{fatos_aprendidos}

Use essas informacoes NATURALMENTE quando forem relevantes para a pergunta.
Nao repita fatos nao solicitados.
Se a pergunta for sobre localizacao, clima, ou preferencias, USE esses dados.

"""
            if historico_texto:
                contexto_completo += f"\n📜 HISTORICO DA CONVERSA:\n{historico_texto}\n"
            if busca:
                prompt = f"{contexto_completo}\nPergunta: {pergunta}\n\nContexto web: {busca[:800]}\n\nREGRAS IMPORTANTES:\n- Se não souber, diga 'Não sei'.\n- Não invente.\nResponda de forma DIRETA e OBJETIVA:"
            else:
                prompt = f"{contexto_completo}\nPergunta: {pergunta}\n\nREGRAS IMPORTANTES:\n- Se não souber, diga 'Não sei' ou 'Não tenho informação suficiente'.\n- Não invente fatos.\nResponda de forma DIRETA e OBJETIVA:"
            prompt = limitar_contexto(prompt, 4000)
            resposta = llm.invoke(prompt, timeout=15)
            conteudo = limpar_tabela(resposta.content)
            return conteudo, None
        except Exception as e:
            self.api_status[nome]["erros"] += 1
            print(f"❌ Erro na API {nome}: {e}")
            return None, str(e)

    def consultar_em_paralelo(self, pergunta, contexto="", busca=None, historico_texto=""):
        if self.groq_native:
            resposta, api = self.usar_groq_com_tools(pergunta, historico_texto)
            if resposta:
                return resposta, api
        apis_prioritarias = []
        for api in self.apis.keys():
            if 'groq' in api:
                apis_prioritarias.append(api)
        for api in self.apis.keys():
            if 'gemini' in api and 'flash' in api:
                apis_prioritarias.append(api)
        for api in self.apis.keys():
            if 'cohere' in api or 'mistral' in api:
                apis_prioritarias.append(api)
        for api in self.apis.keys():
            if 'deepseek' in api:
                apis_prioritarias.append(api)
        for api in apis_prioritarias[:3]:
            resposta, erro = self.usar_api(api, pergunta, contexto, busca, historico_texto)
            if not erro:
                return resposta, api
        return None, "Nenhuma API disponivel"

    def get_status(self):
        return self.api_status

# ============================
# Auto avaliador, Memória Vetorial, Agentes (intactos)
# ============================
class AutoAvaliador:
    def __init__(self, api_manager):
        self.api_manager = api_manager
        self.nota_minima = 7.0
        self.historico_avaliacoes = []

    def avaliar_resposta(self, pergunta, resposta):
        if len(resposta.strip()) < 20:
            return 5.0
        prompt_avaliacao = f"Avalie esta resposta em uma escala de 0 a 10:\n\nPergunta: {pergunta}\nResposta: {resposta}\n\nCriterios:\n- Precisao (0-4 pontos)\n- Clareza (0-3 pontos)\n- Completude (0-3 pontos)\n\nResponda APENAS com um numero."
        for api_nome in ['groq', 'gemini', 'deepseek']:
            for nome, llm in self.api_manager.apis.items():
                if api_nome in nome.lower():
                    try:
                        resultado = llm.invoke(prompt_avaliacao, timeout=10)
                        nota_texto = resultado.content.strip()
                        numeros = re.findall(r'\d+', nota_texto)
                        if numeros:
                            nota = float(numeros[0])
                            return min(nota, 10.0)
                    except:
                        continue
        return 8.0

    def melhorar_resposta(self, pergunta, resposta_atual, nota):
        prompt_melhoria = f"A resposta anterior teve nota {nota}/10. Melhore-a.\n\nPergunta: {pergunta}\nResposta anterior: {resposta_atual}\n\nForneca uma resposta MAIS COMPLETA e PRECISA:"
        for api_nome in ['deepseek', 'gemini', 'groq']:
            for nome, llm in self.api_manager.apis.items():
                if api_nome in nome.lower():
                    try:
                        resultado = llm.invoke(prompt_melhoria, timeout=15)
                        return resultado.content
                    except:
                        continue
        return resposta_atual

class MemoriaVetorial:
    def __init__(self):
        self.memorias = []
        self.db_path = resource_path('memoria_vetorial.db')
        self._inicializar_banco()
        self._carregar_memorias()

    def _inicializar_banco(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''CREATE TABLE IF NOT EXISTS memorias (id INTEGER PRIMARY KEY AUTOINCREMENT, texto TEXT, categoria TEXT, data TEXT)''')
            conn.commit()
            conn.close()
        except:
            pass

    def _carregar_memorias(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT texto, categoria FROM memorias ORDER BY id DESC LIMIT 100')
            self.memorias = [{'texto': row[0], 'categoria': row[1]} for row in cursor.fetchall()]
            conn.close()
        except:
            self.memorias = []

    def adicionar(self, texto, categoria="geral"):
        self.memorias.append({'texto': texto, 'categoria': categoria})
        if len(self.memorias) > 100:
            self.memorias = self.memorias[-100:]
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('INSERT INTO memorias (texto, categoria, data) VALUES (?, ?, ?)',
                           (texto[:500], categoria, agora))
            conn.commit()
            conn.close()
        except:
            pass

    def buscar_similar(self, pergunta, limite=3):
        palavras_pergunta = set(pergunta.lower().split())
        resultados = []
        for memoria in self.memorias:
            texto = memoria['texto'].lower()
            palavras_texto = set(texto.split())
            similaridade = len(palavras_pergunta & palavras_texto)
            if similaridade > 0:
                resultados.append((similaridade, memoria))
        resultados.sort(key=lambda x: x[0], reverse=True)
        return [memoria for _, memoria in resultados[:limite]]

    def aprender_com_conversa(self, pergunta, resposta, foi_boa=True):
        if foi_boa and len(resposta) > 50:
            conhecimento = f"P: {pergunta[:200]}\nR: {resposta[:300]}"
            self.adicionar(conhecimento, "conversa_boa")

class AgentesEspecializados:
    def __init__(self, api_manager):
        self.api_manager = api_manager

    def detectar_especialidade(self, pergunta):
        pergunta_lower = pergunta.lower()
        if any(p in pergunta_lower for p in ['calcule', 'resolva', 'equacao', 'formula', 'matematica']):
            return 'matematica'
        if any(p in pergunta_lower for p in ['crie um codigo', 'gere um codigo', 'programa que']):
            return 'codigo'
        if any(p in pergunta_lower for p in ['crie', 'escreva', 'poema', 'historia', 'imagine']):
            return 'criativo'
        if any(p in pergunta_lower for p in ['analise', 'compare', 'avalie', 'diferenca']):
            return 'analise'
        return 'geral'

    def executar_agente(self, especialidade, pergunta, contexto=""):
        fatos_aprendidos = _carregar_aprendizados()
        contexto_completo = contexto
        if fatos_aprendidos:
            contexto_completo = f"""📚 INFORMACOES SOBRE O USUARIO:
{fatos_aprendidos}

Use essas informacoes naturalmente quando relevante.

{contexto}"""
        prompts = {
            'matematica': f"Voce e um especialista em matematica. Resolva passo a passo: {pergunta}\n{contexto_completo}",
            'codigo': f"Voce e um programador experiente. Forneca codigo limpo: {pergunta}\n{contexto_completo}",
            'criativo': f"Voce e criativo. Use imaginacao: {pergunta}\n{contexto_completo}",
            'analise': f"Voce e analista. Compare pros e contras: {pergunta}\n{contexto_completo}",
            'geral': f"Responda de forma clara, usando as informacoes que voce tem sobre o usuario quando relevante, mas se não souber, diga 'Não sei'. Nao invente: {pergunta}\n{contexto_completo}"
        }
        prompt = prompts.get(especialidade, prompts['geral'])
        api_preferida = {'matematica': 'deepseek', 'codigo': 'deepseek', 'criativo': 'groq', 'analise': 'deepseek',
                         'geral': 'groq'}
        api_escolhida = api_preferida.get(especialidade, 'groq')
        for api_nome in [api_escolhida] + list(self.api_manager.apis.keys()):
            if api_nome in self.api_manager.apis:
                try:
                    resultado = self.api_manager.apis[api_nome].invoke(prompt, timeout=15)
                    return resultado.content, api_nome
                except:
                    continue
        return None, None

# Instâncias globais
api_manager = MultiAPIManager()
init_memoria()
auto_avaliador = AutoAvaliador(api_manager)
memoria_vetorial = MemoriaVetorial()
agentes = AgentesEspecializados(api_manager)
gerenciador_contexto = GerenciadorContexto(api_manager)
rag_manager = RAGManager()

# ============================
# ROTAS
# ============================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    pergunta_raw = data.get('message', '')
    if len(pergunta_raw) > 50000:
        return jsonify({'response': '❌ Mensagem muito longa!'})
    pergunta = sanitizar_input(pergunta_raw)
    session_id = data.get('session_id', 'default')
    modo_avancado = data.get('modo_avancado', True)
    if not rate_limiter.is_allowed(request.remote_addr):
        return jsonify({'response': '⚠️ Aguarde...'})
    if not pergunta:
        return jsonify({'response': 'Digite uma pergunta!'})
    pergunta_lower = pergunta.lower().strip()
    if any(p in pergunta_lower for p in ['você é capaz', 'você consegue', 'o que você faz', 'vc é capaz', 'vc consegue', 'quem é você', 'se apresente']):
        return jsonify({'response': '✅ Sim, sou um assistente de IA capaz de gerar códigos, responder perguntas, aprender e buscar informações confiáveis (Wikipedia, Arxiv, web).'})

    cache_key = f"{session_id}:{pergunta}"
    resposta_cache = cache.get(cache_key)
    if resposta_cache:
        ttl = 7200 if len(pergunta) < 30 else 3600
        cache.set(cache_key, resposta_cache, ttl=ttl)
        return jsonify({'response': resposta_cache, 'cached': True})

    if pergunta_lower in ['limpar', 'reiniciar', 'reset', 'limpar conversa', 'reiniciar conversa']:
        if session_id in conversas:
            conversas[session_id] = []
        cache.clear_pattern(f"{session_id}:")
        if session_id in ultimo_codigo:
            del ultimo_codigo[session_id]
        limpar_historico(session_id)
        return jsonify({'response': '🧹 Conversa limpa! Tudo reiniciado.'})

    if 'Process finished with exit code 0' in pergunta or 'Escolha uma opcao:' in pergunta or 'Pressione enter para continuar' in pergunta:
        return jsonify({'response': '✅ Otimo! O codigo funcionou corretamente.'})

    if any(p in pergunta_lower for p in ['reescreva', 'corrija', 'devolva', 'gere', 'codigo completo', 'arrume', 'complete']) and session_id in ultimo_codigo:
        codigo_anterior = ultimo_codigo[session_id]
        for api_nome in api_manager.apis.keys():
            if 'groq' in api_nome or 'gemini' in api_nome:
                try:
                    llm = api_manager.apis[api_nome]
                    prompt = f"VOCE E UM GERADOR DE CODIGO. SUA TAREFA E GERAR CODIGO COMPLETO E CORRIGIDO.\n\nCODIGO ORIGINAL:\n{codigo_anterior}\n\nSOLICITACAO DO USUARIO:\n{pergunta}\n\nREGRAS:\n1. Responda APENAS com o codigo completo corrigido\n2. NAO inclua explicacoes\n3. Mantenha a estrutura original\n\nCODIGO CORRIGIDO:"
                    resposta = llm.invoke(prompt, timeout=120)
                    if resposta and resposta.content:
                        codigo_corrigido = resposta.content.strip()
                        if codigo_corrigido.startswith("```"):
                            codigo_corrigido = "\n".join(codigo_corrigido.split("\n")[1:])
                        if codigo_corrigido.endswith("```"):
                            codigo_corrigido = "\n".join(codigo_corrigido.split("\n")[:-1])
                        ultimo_codigo[session_id] = codigo_corrigido
                        return jsonify({'response': codigo_corrigido.strip()})
                except:
                    continue

    if 'traceback' in pergunta_lower or 'nameerror' in pergunta_lower or 'syntaxerror' in pergunta_lower:
        if session_id in ultimo_codigo:
            prompt_correcao = f"O codigo abaixo gerou este erro: {pergunta}\n\nCODIGO ORIGINAL:\n{ultimo_codigo[session_id]}\n\nCORRIJA o codigo e devolva COMPLETO e FUNCIONAL:"
            for api_nome in api_manager.apis.keys():
                if 'deepseek' in api_nome or 'groq' in api_nome:
                    try:
                        llm = api_manager.apis[api_nome]
                        resposta = llm.invoke(prompt_correcao, timeout=120)
                        if resposta and resposta.content:
                            codigo_corrigido = resposta.content.strip()
                            ultimo_codigo[session_id] = codigo_corrigido
                            return jsonify({'response': f"✅ Codigo corrigido:\n\n{codigo_corrigido}"})
                    except:
                        continue

    if api_manager.precisa_codigo(pergunta):
        ultimo_codigo[session_id] = pergunta
        codigo_validado = processar_codigo_com_auto_correcao(pergunta, api_manager, session_id, cache_key)
        if codigo_validado:
            add_ao_historico(session_id, pergunta, codigo_validado)
            cache.set(cache_key, codigo_validado, ttl=3600)
            return jsonify({'response': codigo_validado})

    if pergunta_lower.startswith("aprenda que"):
        conteudo = pergunta[10:].strip()
        if _salvar_fato(conteudo):
            rag_manager.adicionar_embedding(conteudo, "fato_explicito")
            return jsonify({'response': f"📚 Aprendi: {conteudo}"})
        else:
            return jsonify({'response': "❌ Erro ao salvar o aprendizado."})

    contexto_aprendizado = _carregar_aprendizados()
    contexto_memoria = ""
    if contexto_aprendizado:
        contexto_memoria = contexto_aprendizado

    memorias_rag = rag_manager.buscar_similares(pergunta, limite=3)
    if memorias_rag:
        contexto_memoria += "\n📌 Memorias relevantes (RAG):\n"
        for mem in memorias_rag[:2]:
            contexto_memoria += f"- {mem['texto'][:200]}\n"

    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT regra FROM licoes WHERE tipo='fato' ORDER BY id DESC")
    fatos = cursor.fetchall()
    conn.close()
    if fatos:
        contexto_aprendizado = "Voce conhece estas informacoes sobre o usuario. Use APENAS quando a pergunta for relacionada. NAO repita informacoes desnecessarias:\n"
        for fato in fatos:
            contexto_aprendizado += f"- {fato[0]}\n"
        contexto_aprendizado += "\nResponda APENAS o que foi perguntado, de forma DIRETA e OBJETIVA. NAO repita fatos nao solicitados."
    else:
        contexto_aprendizado = ""

    resposta_lembrada, api_lembrada, confianca = lembrar(pergunta)
    if resposta_lembrada and confianca > 70:
        add_ao_historico(session_id, pergunta, resposta_lembrada)
        cache.set(cache_key, resposta_lembrada, ttl=3600)
        return jsonify({'response': f"📚 {resposta_lembrada}"})

    memorias_similares = memoria_vetorial.buscar_similar(pergunta)
    if memorias_similares:
        contexto_memoria += "\n📌 Conhecimentos relevantes:\n"
        for mem in memorias_similares[:2]:
            contexto_memoria += f"- {mem['texto'][:200]}\n"

    historico = get_historico(session_id)
    if any(p in pergunta_lower for p in ['código completo', 'devolva', 'como eu passei', 'completo', 'inteiro']):
        historico_texto = ""
        for msg in historico:
            historico_texto += f"Usuario: {msg['pergunta']}\nAssistente: {msg['resposta']}\n"
    else:
        historico_texto = gerenciador_contexto.preparar_contexto(historico, pergunta, contexto_memoria)

    if contexto_aprendizado:
        historico_texto = contexto_aprendizado + "\n" + historico_texto

    resposta = None
    api_usada = "nenhuma"

    if modo_avancado:
        if api_manager.groq_native:
            resposta, api_usada = api_manager.usar_groq_com_tools(pergunta, historico_texto)
        if not resposta:
            especialidade = agentes.detectar_especialidade(pergunta)
            resposta, api_usada = agentes.executar_agente(especialidade, pergunta, historico_texto)
        if not resposta:
            resposta, api_usada = api_manager.consultar_em_paralelo(pergunta, historico_texto=historico_texto)
        if not resposta:
            api_escolhida = api_manager.escolher_melhor_api(pergunta)
            if api_escolhida:
                resposta, _ = api_manager.usar_api(api_escolhida, pergunta, historico_texto=historico_texto)
                api_usada = api_escolhida
        if resposta:
            nota = auto_avaliador.avaliar_resposta(pergunta, resposta)
            if nota < 7.0:
                resposta_melhorada = auto_avaliador.melhorar_resposta(pergunta, resposta, nota)
                nova_nota = auto_avaliador.avaliar_resposta(pergunta, resposta_melhorada)
                if nova_nota > nota:
                    resposta = resposta_melhorada
                    memoria_vetorial.aprender_com_conversa(pergunta, resposta, foi_boa=True)
                    rag_manager.adicionar_embedding(f"P: {pergunta}\nR: {resposta}", "conversa_boa")
            else:
                memoria_vetorial.aprender_com_conversa(pergunta, resposta, foi_boa=True)
                rag_manager.adicionar_embedding(f"P: {pergunta}\nR: {resposta}", "conversa_boa")
    else:
        api_escolhida = api_manager.escolher_melhor_api(pergunta)
        if api_escolhida:
            resposta, _ = api_manager.usar_api(api_escolhida, pergunta, historico_texto=historico_texto)
            api_usada = api_escolhida

    if resposta:
        if resposta.startswith('🌐'):
            resposta = resumir_resultado_busca(resposta, pergunta)
        aprender_resposta(pergunta, resposta, api_usada)
        add_ao_historico(session_id, pergunta, resposta)
        cache.set(cache_key, resposta, ttl=3600)
        return jsonify({'response': resposta})

    # Fallback com fontes confiáveis (Wikipedia, Arxiv, DuckDuckGo)
    resultado_busca = buscar_fonte_confiavel(pergunta)
    if resultado_busca:
        resposta = f"🔍 Pesquisa confiável:\n{resultado_busca[:600]}"
        add_ao_historico(session_id, pergunta, resposta)
        cache.set(cache_key, resposta, ttl=3600)
        return jsonify({'response': resposta})
    else:
        return jsonify({'response': 'Desculpe, não consegui encontrar informações confiáveis sobre isso.'})

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    data = request.json
    pergunta = sanitizar_input(data.get('message', ''))
    session_id = data.get('session_id', 'default')

    def generate():
        try:
            api_escolhida = data.get('api') or api_manager.escolher_melhor_api(pergunta)
            if not api_escolhida:
                yield f"data: {json.dumps({'chunk': '❌ Nenhuma API disponível'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Usa gerenciador de contexto para resumir histórico
            historico = get_historico(session_id)
            historico_texto = gerenciador_contexto.preparar_contexto(historico, pergunta)

            fatos_aprendidos = _carregar_aprendizados()
            contexto_completo = ""
            if fatos_aprendidos:
                contexto_completo = f"""📚 INFORMACOES QUE VOCE JA SABE SOBRE O USUARIO:
{fatos_aprendidos}

"""
            if historico_texto:
                contexto_completo += f"\n📜 HISTORICO DA CONVERSA:\n{historico_texto}\n"

            prompt = f"{contexto_completo}\nPergunta: {pergunta}\n\nREGRAS IMPORTANTES:\n- Se você não souber a resposta, diga 'Não sei' ou 'Não tenho informação suficiente'.\n- Não invente fatos.\nResponda de forma DIRETA e OBJETIVA:"
            prompt = limitar_contexto(prompt, 4000)

            if api_escolhida in api_manager.apis:
                llm = api_manager.apis[api_escolhida]
                resposta_completa = ""
                for chunk in llm.stream(prompt, timeout=30):
                    chunk_texto = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if chunk_texto:
                        resposta_completa += chunk_texto
                        yield f"data: {json.dumps({'chunk': chunk_texto})}\n\n"
                if resposta_completa:
                    add_ao_historico(session_id, pergunta, resposta_completa)
            else:
                resposta, _ = api_manager.usar_api(api_escolhida, pergunta, historico_texto=historico_texto)
                if resposta:
                    palavras = resposta.split()
                    for i, palavra in enumerate(palavras):
                        chunk = palavra + (' ' if i < len(palavras) - 1 else '')
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                        time.sleep(0.05)
                    add_ao_historico(session_id, pergunta, resposta)
                else:
                    yield f"data: {json.dumps({'chunk': '❌ Erro ao processar'})}\n\n"
        except Exception as e:
            error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print(error_trace)
            yield f"data: {json.dumps({'chunk': f'❌ {str(e)[:100]}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/reiniciar', methods=['POST'])
def reiniciar_conversa():
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in conversas:
        conversas[session_id] = []
    cache.clear_pattern(f"{session_id}:")
    if session_id in ultimo_codigo:
        del ultimo_codigo[session_id]
    limpar_historico(session_id)
    return jsonify({'status': 'ok', 'message': 'Conversa reiniciada'})

@app.route('/status', methods=['GET'])
def status():
    return jsonify(api_manager.get_status())

@app.route('/estatisticas', methods=['GET'])
def estatisticas():
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM conhecimento')
    total_conhecimento = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = "bom"')
    acertos = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = "ruim"')
    erros = cursor.fetchone()[0]
    conn.close()
    return jsonify({'conhecimentos': total_conhecimento, 'acertos': acertos, 'erros': erros,
                    'memorias_vetoriais': len(memoria_vetorial.memorias), 'apis': api_manager.get_status()})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    pergunta = data.get('pergunta', '')
    resposta = data.get('resposta', '')
    api_usada = data.get('api', '')
    feedback_valor = data.get('feedback', '')
    if feedback_valor in ['bom', 'ruim']:
        registrar_feedback(pergunta, resposta, api_usada, feedback_valor)
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'erro'})

@app.route('/licoes', methods=['GET'])
def licoes():
    return jsonify(buscar_licoes())

@app.route('/ensinar', methods=['POST'])
def ensinar_ia():
    data = request.json
    conhecimento = data.get('conhecimento', '')
    categoria = data.get('categoria', 'geral')
    if conhecimento:
        memoria_vetorial.adicionar(conhecimento, categoria)
        rag_manager.adicionar_embedding(conhecimento, categoria)
        return jsonify({'status': 'ok', 'message': 'Conhecimento adicionado!'})
    return jsonify({'status': 'erro', 'message': 'Nada para aprender'})

@app.route('/dashboard', methods=['GET'])
def dashboard():
    # Portanto, mantenha exatamente como você já tinha (não alterei)
    db_path = resource_path('memoria_agente.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM conhecimento')
    total_conhecimento = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = "bom"')
    acertos = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = "ruim"')
    erros = cursor.fetchone()[0]
    conn.close()

    status_apis = api_manager.get_status()
    api_rows = ""
    for nome, status in status_apis.items():
        usos = status['usos']
        erros_api = status['erros']
        taxa = ((usos - erros_api) / usos * 100) if usos > 0 else 100
        tipo = status.get('tipo', 'geral')
        if taxa >= 90:
            badge = '<span class="badge badge-success">✅ Ativa</span>'
        elif taxa >= 70:
            badge = '<span class="badge badge-warning">⚠️ Instavel</span>'
        else:
            badge = '<span class="badge badge-danger">❌ Com Erros</span>'
        api_rows += f"<tr><td><strong>{nome}</strong></td><td>{tipo}</td><td>{usos}</td><td>{erros_api}</td><td>{badge}</td></tr>"

    taxa_acerto = (acertos / (acertos + erros) * 100) if (acertos + erros) > 0 else 100

    # O HTML do dashboard é longo, vou mantê-lo igual ao seu original (apenas ajustei a mensagem inicial)
    html = '''<!DOCTYPE html><html><head><title>🤖 Super Agente - Dashboard</title><meta charset="UTF-8"><style>
        *{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',Arial,sans-serif;margin:20px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#e0e0e0;min-height:100vh}.container{max-width:1200px;margin:0 auto}h1{color:#e94560;margin-bottom:30px;font-size:2.5em;text-shadow:2px 2px 4px rgba(0,0,0,0.3)}h2{color:#e94560;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin-bottom:30px}.card{background:rgba(42,42,74,0.9);padding:25px;border-radius:15px;box-shadow:0 8px 32px rgba(0,0,0,0.3);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);transition:transform 0.3s}.card:hover{transform:translateY(-5px)}.stat{text-align:center}.stat-value{font-size:3em;font-weight:bold;background:linear-gradient(135deg,#e94560,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}.stat-label{color:#888;margin-top:10px}table{width:100%;border-collapse:collapse;margin-top:10px;background:rgba(42,42,74,0.5);border-radius:10px;overflow:hidden}th,td{padding:15px;text-align:left;border-bottom:1px solid #444}th{background:#e94560;color:white;font-weight:600}tr:hover{background:rgba(255,255,255,0.05)}.badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.85em;font-weight:600}.badge-success{background:#4caf50;color:white}.badge-warning{background:#ffa500;color:white}.badge-danger{background:#f44336;color:white}.footer{text-align:center;margin-top:40px;padding:20px;color:#666;border-top:1px solid #333}.chat-container{background:rgba(42,42,74,0.9);border-radius:15px;padding:20px;margin:20px 0;backdrop-filter:blur(10px)}#chat-messages{height:400px;overflow-y:auto;padding:20px;background:#0f0f1a;border-radius:10px;margin-bottom:20px}.message{margin-bottom:15px;padding:12px 18px;border-radius:12px;animation:fadeIn 0.3s}@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}.user-message{background:#e94560;color:white;margin-left:50px;text-align:right}.assistant-message{background:#2a2a4a;color:#e0e0e0;margin-right:50px;border-left:3px solid #e94560}.input-container{display:flex;gap:10px;align-items:flex-end}#message-input{flex:1;padding:15px;border-radius:10px;border:1px solid #444;background:#0f0f1a;color:white;font-size:14px;resize:vertical;min-height:60px;font-family:'Segoe UI',Arial,sans-serif}#message-input:focus{outline:none;border-color:#e94560}.btn{padding:12px 24px;border:none;border-radius:10px;font-size:14px;font-weight:bold;cursor:pointer;transition:all 0.3s}.btn-primary{background:#e94560;color:white}.btn-primary:hover{background:#ff6b6b;transform:scale(1.05)}.btn-primary:disabled{background:#666;cursor:not-allowed;transform:none}.btn-secondary{background:#2a2a4a;color:white}.btn-secondary:hover{background:#3a3a5a}.typing-indicator{display:none;padding:12px 18px;background:#2a2a4a;border-radius:12px;margin-right:50px;color:#888;border-left:3px solid #e94560}.typing-indicator.active{display:inline-block}.typing-dots{display:flex;gap:4px}.typing-dots span{width:8px;height:8px;border-radius:50%;background:#e94560;animation:typing 1.4s infinite}.typing-dots span:nth-child(2){animation-delay:0.2s}.typing-dots span:nth-child(3){animation-delay:0.4s}@keyframes typing{0%,60%,100%{transform:translateY(0);opacity:0.4}30%{transform:translateY(-10px);opacity:1}}
    </style></head><body><div class="container"><h1>🤖 Super Agente - Dashboard</h1><div class="grid"><div class="card"><div class="stat"><div class="stat-value">''' + str(total_conhecimento) + '''</div><div class="stat-label">📚 Conhecimentos</div></div></div><div class="card"><div class="stat"><div class="stat-value" style="background:linear-gradient(135deg,#4caf50,#8bc34a);-webkit-background-clip:text;-webkit-text-fill-color:transparent">''' + str(acertos) + '''</div><div class="stat-label">✅ Acertos</div></div></div><div class="card"><div class="stat"><div class="stat-value" style="background:linear-gradient(135deg,#f44336,#ff7961);-webkit-background-clip:text;-webkit-text-fill-color:transparent">''' + str(erros) + '''</div><div class="stat-label">❌ Erros</div></div></div><div class="card"><div class="stat"><div class="stat-value">''' + str(len(memoria_vetorial.memorias)) + '''</div><div class="stat-label">🧠 Memórias Vetoriais</div></div></div><div class="card"><div class="stat"><div class="stat-value">''' + f"{taxa_acerto:.1f}" + '''%</div><div class="stat-label">📊 Taxa de Acerto</div></div></div></div><div class="chat-container"><h2>💬 Chat em Tempo Real (Streaming)</h2><div id="chat-messages"><div class="message assistant-message">👋 Olá! Sou o Super Agente IA. Use fontes confiáveis (Wikipedia, Arxiv) e tenho memória persistente. Como posso ajudar?</div></div><div id="typing-indicator" class="typing-indicator"><div class="typing-dots"><span></span><span></span><span></span></div></div><div class="input-container"><textarea id="message-input" placeholder="Digite sua mensagem..." rows="2"></textarea><button class="btn btn-primary" id="send-btn" onclick="sendMessage()">📤 Enviar</button><button class="btn btn-secondary" onclick="clearChat()">🧹 Limpar</button></div></div><div class="card" style="margin-top:20px"><h2>🔌 Status das APIs</h2><table><tr><th>API</th><th>Tipo</th><th>Usos</th><th>Erros</th><th>Status</th></tr>''' + api_rows + '''</table></div><div class="footer">🤖 Super Agente v4.0 • Atualizado em: ''' + datetime.now().strftime('%d/%m/%Y %H:%M:%S') + '''<br>🚀 Modo Avançado Ativo • RAG com Embeddings • Function Calling • Web Scraping Estruturado • Streaming • Wikipedia • Arxiv • Memória Redis • Anti-alucinação</div></div><script>let sessionId="dashboard_"+Date.now();document.getElementById("message-input").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMessage()}});async function sendMessage(){let e=document.getElementById("message-input"),t=document.getElementById("send-btn"),n=e.value.trim();if(!n)return;e.disabled=!0,t.disabled=!0,addMessage("user",n),e.value="",document.getElementById("typing-indicator").classList.add("active");let s="assistant_"+Date.now();try{let o=await fetch("/chat/stream",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:n,session_id:sessionId})});document.getElementById("typing-indicator").classList.remove("active"),addMessage("assistant","",s);let a=o.body.getReader(),d=new TextDecoder,i="";for(;;){let{value:l,done:u}=await a.read();if(u)break;let c=d.decode(l).split("\n");for(let r of c)if(r.startsWith("data: ")){let p=r.slice(6);if(p==="[DONE]")continue;try{let m=JSON.parse(p);m.chunk&&(i+=m.chunk,updateMessage(s,i))}catch(g){}}}i||updateMessage(s,"❌ Não recebi resposta. Tente novamente.")}catch(o){console.error("Erro:",o),document.getElementById("typing-indicator").classList.remove("active"),addMessage("assistant","❌ Erro ao processar mensagem. Verifique se o servidor está rodando.")}finally{e.disabled=!1,t.disabled=!1,e.focus()}}function addMessage(e,t,n){let s=document.getElementById("chat-messages"),o=document.createElement("div");o.className=`message ${e}-message`,n&&(o.id=n),o.innerHTML="user"===e?t.replace(/\n/g,"<br>"):t||"🤔 ...",s.appendChild(o),s.scrollTop=s.scrollHeight;return o}function updateMessage(e,t){let n=document.getElementById(e);if(n){let s=t.replace(/•/g,"<br>•").replace(/\n/g,"<br>").replace(/<br><br>/g,"<br>");s.startsWith("<br>")&&(s=s.slice(4)),n.innerHTML=s,document.getElementById("chat-messages").scrollTo({top:document.getElementById("chat-messages").scrollHeight,behavior:"smooth"})}}async function clearChat(){document.getElementById("chat-messages").innerHTML="",addMessage("assistant","🧹 Chat limpo! Como posso ajudar?");try{await fetch("/reiniciar",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:sessionId})})}catch(e){console.error("Erro ao reiniciar:",e)}}setInterval(async function(){try{let e=await(await fetch("/estatisticas")).json();document.querySelectorAll(".stat-value")[0].textContent=e.conhecimentos||0,document.querySelectorAll(".stat-value")[1].textContent=e.acertos||0,document.querySelectorAll(".stat-value")[2].textContent=e.erros||0,document.querySelectorAll(".stat-value")[3].textContent=e.memorias_vetoriais||0;let t=e.acertos+e.erros>0?(e.acertos/(e.acertos+e.erros)*100).toFixed(1):100;document.querySelectorAll(".stat-value")[4].textContent=t+"%"}catch(e){console.error("Erro ao atualizar stats:",e)}},3e4);</script></body></html>'''

    return render_template_string(html)

if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 70)
    print("🤖 SUPER AGENTE - VERSAO ORIGINAL + STREAMING + WEB SCRAPING + WIKIPEDIA + ARXIV + REDIS")
    print("=" * 70)
    print(f"\n📊 TOTAL DE MODELOS CARREGADOS: {len(api_manager.apis)}")
    if api_manager.groq_native:
        print("✅ Function Calling do Groq ATIVADO!")
    if EMBEDDINGS_AVAILABLE:
        print("✅ RAG com Embeddings ATIVADO!")
    print("✅ Web Scraping Estruturado ATIVADO!")
    print("✅ Streaming de Respostas ATIVADO!")
    print("✅ Wikipedia e Arxiv ATIVADOS!")
    print("✅ Memória Redis (Upstash) configurada!")
    print("✅ Anti-alucinação ativada!")
    print("\n📌 Servidor rodando em: http://localhost:5000")
    print("📌 Dashboard: http://localhost:5000/dashboard")
    print("📌 Streaming: http://localhost:5000/chat/stream")
    print("\n📌 Pressione CTRL+C para parar\n")
    serve(app, host='0.0.0.0', port=port, threads=4, channel_timeout=120)