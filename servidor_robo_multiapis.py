import os
import json
import requests
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain base
from langchain_community.tools import DuckDuckGoSearchRun

# APIs (apenas as estáveis)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_mistralai.chat_models import ChatMistralAI

load_dotenv()

app = Flask(__name__)
CORS(app)


# ============================================
# CONFIGURAÇÃO DAS APIS (VERSÃO ESTÁVEL)
# ============================================

class MultiAPIManager:
    def __init__(self):
        self.apis = {}
        self.api_status = {}
        self.carregar_apis()

    def carregar_apis(self):
        """Carrega todas as APIs disponíveis (apenas as estáveis)"""

        # 1. Google Gemini (Estável)
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self.apis["gemini"] = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.3,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                self.api_status["gemini"] = {"status": "ok", "usos": 0, "erros": 0}
                print("✅ Gemini (Google) carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Gemini: {e}")

        # 2. Groq (Mais rápido e estável)
        if os.getenv("GROQ_API_KEY"):
            try:
                self.apis["groq"] = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.3
                )
                self.api_status["groq"] = {"status": "ok", "usos": 0, "erros": 0}
                print("✅ Groq carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Groq: {e}")

        # 3. DeepSeek (Ótimo para raciocínio)
        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                self.apis["deepseek"] = ChatOpenAI(
                    model="deepseek-chat",
                    openai_api_base="https://api.deepseek.com/v1",
                    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                    temperature=0.3
                )
                self.api_status["deepseek"] = {"status": "ok", "usos": 0, "erros": 0}
                print("✅ DeepSeek carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar DeepSeek: {e}")

        # 4. OpenAI (ChatGPT) - Backup confiável
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.apis["openai"] = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0.3
                )
                self.api_status["openai"] = {"status": "ok", "usos": 0, "erros": 0}
                print("✅ OpenAI carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar OpenAI: {e}")

        # 5. Cohere (Bom para busca)
        if os.getenv("COHERE_API_KEY"):
            try:
                self.apis["cohere"] = ChatCohere(
                    model="command-r",
                    cohere_api_key=os.getenv("COHERE_API_KEY"),
                    temperature=0.3
                )
                self.api_status["cohere"] = {"status": "ok", "usos": 0, "erros": 0}
                print("✅ Cohere carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Cohere: {e}")

        # 6. Mistral AI (Alternativa europeia)
        if os.getenv("MISTRAL_API_KEY"):
            try:
                self.apis["mistral"] = ChatMistralAI(
                    model="mistral-large-latest",
                    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                    temperature=0.3
                )
                self.api_status["mistral"] = {"status": "ok", "usos": 0, "erros": 0}
                print("✅ Mistral carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar Mistral: {e}")

        # 7. DeepSeek Reasoner (versão com raciocínio explícito)
        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                self.apis["deepseek_reasoner"] = ChatOpenAI(
                    model="deepseek-reasoner",
                    openai_api_base="https://api.deepseek.com/v1",
                    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                    temperature=0.3
                )
                self.api_status["deepseek_reasoner"] = {"status": "ok", "usos": 0, "erros": 0}
                print("✅ DeepSeek Reasoner carregado!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar DeepSeek Reasoner: {e}")

        if not self.apis:
            print("⚠️ NENHUMA API carregada! Verifique suas chaves no arquivo .env")
            print("   Certifique-se de ter pelo menos uma das chaves: GEMINI, GROQ, DEEPSEEK, OPENAI")

    def escolher_melhor_api(self, pergunta):
        """Escolhe a melhor API baseado no tipo de pergunta"""
        pergunta_lower = pergunta.lower()

        # Raciocínio complexo -> DeepSeek Reasoner ou DeepSeek
        if any(palavra in pergunta_lower for palavra in
               ['calcule', 'resolva', 'matemática', 'lógica', 'física', 'química', 'porque',
                'explique detalhadamente']):
            if "deepseek_reasoner" in self.apis:
                return "deepseek_reasoner"
            if "deepseek" in self.apis:
                return "deepseek"

        # Respostas rápidas e curtas -> Groq (mais rápido)
        if len(pergunta.split()) < 10:
            if "groq" in self.apis:
                return "groq"

        # Perguntas longas ou contexto grande -> Gemini
        if len(pergunta) > 500:
            if "gemini" in self.apis:
                return "gemini"

        # Conversas gerais -> a que tiver disponível
        # Ordem de preferência: groq, gemini, deepseek, openai, cohere, mistral
        ordem_preferencia = ["groq", "gemini", "deepseek", "openai", "cohere", "mistral", "deepseek_reasoner"]

        for nome_api in ordem_preferencia:
            if nome_api in self.apis:
                return nome_api

        # Se nenhuma da lista estiver disponível, pega a primeira
        if self.apis:
            return list(self.apis.keys())[0]

        return None

    def usar_api(self, nome, pergunta, contexto=""):
        """Usa uma API específica"""
        if nome not in self.apis:
            return None, f"API {nome} não disponível"

        try:
            self.api_status[nome]["usos"] += 1

            llm = self.apis[nome]

            prompt = f"""
            Contexto anterior: {contexto}

            Pergunta do usuário: {pergunta}

            Responda de forma clara, direta e amigável em português brasileiro.
            Seja objetivo e use emojis quando apropriado.
            """

            resposta = llm.invoke(prompt)
            return resposta.content, None

        except Exception as e:
            self.api_status[nome]["erros"] += 1
            print(f"❌ Erro na API {nome}: {e}")
            return None, str(e)

    def get_status(self):
        """Retorna status de todas as APIs"""
        return self.api_status


# ============================================
# INICIALIZAÇÃO
# ============================================

# Carrega APIs
api_manager = MultiAPIManager()

# Ferramenta de busca
search = DuckDuckGoSearchRun()


# Banco de dados de memória
def init_memoria():
    conn = sqlite3.connect('memoria_agente.db')
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
    conn.commit()
    conn.close()


def aprender_resposta(pergunta, resposta, api_usada):
    conn = sqlite3.connect('memoria_agente.db')
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
    conn = sqlite3.connect('memoria_agente.db')
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
    """Registra feedback do usuário"""
    conn = sqlite3.connect('memoria_agente.db')
    cursor = conn.cursor()
    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute('''
        INSERT INTO feedback (pergunta, resposta, api_usada, feedback, data)
        VALUES (?, ?, ?, ?, ?)
    ''', (pergunta, resposta, api_usada, feedback_usuario, agora))

    if feedback_usuario == 'bom':
        cursor.execute('''
            UPDATE conhecimento 
            SET acertos = acertos + 1
            WHERE pergunta = ?
        ''', (pergunta,))
    elif feedback_usuario == 'ruim':
        cursor.execute('''
            UPDATE conhecimento 
            SET erros = erros + 1
            WHERE pergunta = ?
        ''', (pergunta,))

    conn.commit()
    conn.close()


# ============================================
# FUNÇÕES DE BUSCA ESPECÍFICAS
# ============================================

def get_bitcoin_price():
    try:
        response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=brl',
                                timeout=5)
        if response.status_code == 200:
            preco = response.json()['bitcoin']['brl']
            return f"R$ {preco:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.')
    except:
        pass
    return None


def get_dollar_price():
    try:
        response = requests.get('https://economia.awesomeapi.com.br/json/last/USD-BRL', timeout=5)
        if response.status_code == 200:
            preco = float(response.json()['USDBRL']['bid'])
            return f"R$ {preco:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.')
    except:
        pass
    return None


def get_euro_price():
    try:
        response = requests.get('https://economia.awesomeapi.com.br/json/last/EUR-BRL', timeout=5)
        if response.status_code == 200:
            preco = float(response.json()['EURBRL']['bid'])
            return f"R$ {preco:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.')
    except:
        pass
    return None


def get_any_crypto_price(moeda):
    moeda = moeda.lower().strip()
    try:
        response = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids={moeda}&vs_currencies=brl',
                                timeout=5)
        if response.status_code == 200 and moeda in response.json():
            preco = response.json()[moeda]['brl']
            return f"R$ {preco:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.')
    except:
        pass
    return None


def get_weather(cidade):
    try:
        response = requests.get(f'https://wttr.in/{cidade}?format=%C+%t', timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except:
        pass
    return None


# ============================================
# ROTA PRINCIPAL
# ============================================

init_memoria()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    pergunta = data.get('message', '')

    if not pergunta:
        return jsonify({'response': 'Por favor, digite uma pergunta!'})

    # Tenta lembrar se já respondeu antes
    resposta_lembrada, api_lembrada, confianca = lembrar(pergunta)

    if resposta_lembrada and confianca > 70:
        resposta = f"📚 Lembrei! (Da última vez usei {api_lembrada})\n\n{resposta_lembrada}"
        return jsonify({'response': resposta})

    # ============================================
    # VERIFICA PERGUNTAS ESPECÍFICAS
    # ============================================

    pergunta_lower = pergunta.lower()

    # Bitcoin
    if "bitcoin" in pergunta_lower and any(p in pergunta_lower for p in ['preço', 'valor', 'quanto', 'preco']):
        preco = get_bitcoin_price()
        if preco:
            return jsonify({'response': f"💰 Bitcoin: {preco}"})

    # Dólar
    if any(p in pergunta_lower for p in ['dólar', 'dolar']) and any(
            p in pergunta_lower for p in ['preço', 'valor', 'quanto', 'cotação', 'preco']):
        preco = get_dollar_price()
        if preco:
            return jsonify({'response': f"💵 Dólar: {preco}"})

    # Euro
    if "euro" in pergunta_lower and any(p in pergunta_lower for p in ['preço', 'valor', 'quanto', 'cotação', 'preco']):
        preco = get_euro_price()
        if preco:
            return jsonify({'response': f"💶 Euro: {preco}"})

    # Criptomoedas em geral
    if any(p in pergunta_lower for p in
           ['cripto', 'ethereum', 'solana', 'bnb', 'ripple', 'cardano', 'dogecoin']) and any(
            p in pergunta_lower for p in ['preço', 'valor', 'quanto']):
        # Tenta extrair o nome da cripto
        criptos_conhecidas = ['ethereum', 'solana', 'bnb', 'ripple', 'cardano', 'dogecoin', 'matic', 'polkadot']
        for cripto in criptos_conhecidas:
            if cripto in pergunta_lower:
                preco = get_any_crypto_price(cripto)
                if preco:
                    return jsonify({'response': f"💰 {cripto.upper()}: {preco}"})

    # Clima
    if "clima" in pergunta_lower or "tempo" in pergunta_lower:
        # Extrai cidade (simplificado)
        palavras = pergunta_lower.split()
        for palavra in palavras:
            if palavra not in ['clima', 'tempo', 'temperatura', 'em', 'na', 'no', 'de', 'como', 'está', 'hoje',
                               'previsão']:
                cidade = palavra
                clima = get_weather(cidade)
                if clima:
                    return jsonify({'response': f"🌡️ Clima em {cidade.title()}: {clima}"})

    # ============================================
    # ESCOLHE A MELHOR API PARA A PERGUNTA
    # ============================================

    api_escolhida = api_manager.escolher_melhor_api(pergunta)

    if not api_escolhida:
        # Fallback: busca na web
        print("⚠️ Nenhuma API disponível! Buscando na web...")
        resultado_busca = search.run(pergunta)
        return jsonify({'response': f"🌐 Busquei na web:\n\n{resultado_busca[:600]}"})

    # Usa a API escolhida
    resposta, erro = api_manager.usar_api(api_escolhida, pergunta)

    # Se a primeira API falhar, tenta as outras em sequência
    if erro:
        print(f"⚠️ API {api_escolhida} falhou: {erro}")
        print("🔄 Tentando outras APIs...")

        for nome_api in api_manager.apis.keys():
            if nome_api != api_escolhida:
                resposta, erro = api_manager.usar_api(nome_api, pergunta)
                if not erro:
                    api_escolhida = nome_api
                    print(f"✅ API {api_escolhida} funcionou como fallback!")
                    break

        # Se todas falharem, usa busca na web
        if erro:
            print("⚠️ Todas as APIs falharam! Usando busca na web...")
            resultado_busca = search.run(pergunta)
            resposta = f"🌐 Busquei na web:\n\n{resultado_busca[:600]}"
            api_escolhida = "busca_web"

    # Aprende a resposta para futuras perguntas
    if resposta and api_escolhida != "busca_web":
        aprender_resposta(pergunta, resposta, api_escolhida)

    return jsonify({'response': resposta})


# ============================================
# ROTAS ADICIONAIS
# ============================================

@app.route('/status', methods=['GET'])
def status():
    """Rota para verificar status das APIs"""
    return jsonify(api_manager.get_status())


@app.route('/estatisticas', methods=['GET'])
def estatisticas():
    """Rota para ver estatísticas de aprendizado"""
    conn = sqlite3.connect('memoria_agente.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM conhecimento')
    total_conhecimento = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = "bom"')
    acertos = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback = "ruim"')
    erros = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        'conhecimentos': total_conhecimento,
        'acertos': acertos,
        'erros': erros,
        'apis': api_manager.get_status()
    })


@app.route('/feedback', methods=['POST'])
def feedback():
    """Rota para registrar feedback do usuário"""
    data = request.json
    pergunta = data.get('pergunta', '')
    resposta = data.get('resposta', '')
    api_usada = data.get('api', '')
    feedback_valor = data.get('feedback', '')

    if feedback_valor in ['bom', 'ruim']:
        registrar_feedback(pergunta, resposta, api_usada, feedback_valor)
        return jsonify({'status': 'ok', 'message': 'Feedback registrado!'})

    return jsonify({'status': 'erro', 'message': 'Feedback inválido'})


# ============================================
# INICIALIZAÇÃO DO SERVIDOR
# ============================================

if __name__ == '__main__':
    import os

    # Pega a porta do ambiente (Render define isso automaticamente)
    port = int(os.environ.get('PORT', 5000))

    print("\n" + "=" * 60)
    print("🤖 SERVIDOR MULTI-API (VERSÃO ESTÁVEL)")
    print("=" * 60)
    print(f"\n📊 APIs carregadas: {len(api_manager.apis)}")

    if api_manager.apis:
        for nome, info in api_manager.apis.items():
            print(f"   ✅ {nome.upper()} - disponível")
    else:
        print("   ⚠️ NENHUMA API carregada!")
        print("   Verifique suas chaves no arquivo .env")

    # Mostra a URL correta baseada no ambiente
    if port == 5000:
        print("\n📌 Abra o arquivo 'interface_bb8.html' no navegador")
        print(f"📌 O robô está ouvindo em: http://localhost:{port}")
    else:
        print(f"\n📌 O robô está rodando na porta: {port}")
        print("📌 Acesse via URL do Render")

    print("📌 Pressione CTRL+C para parar o servidor\n")

    # Host '0.0.0.0' permite acesso externo (necessário para o Render)
    app.run(host='0.0.0.0', port=port, debug=False)