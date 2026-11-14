"""
Detector de Spam em E-mails - Streamlit + TensorFlow

Aplicação corporativa para classificar mensagens de e-mail como SPAM ou
NÃO SPAM. O modelo é uma rede neural simples (TextVectorization + Embedding
+ Dense) treinada na inicialização da aplicação.

Executar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Executar no Render (Start Command):
    streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduz logs do TensorFlow

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Detector de Spam | Marketing",
    page_icon="🛡️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Base de treino de demonstração (rótulo 1 = spam, 0 = não spam)
# Substitua por uma base real da empresa via upload de CSV na barra lateral.
# ---------------------------------------------------------------------------
SPAM_EXEMPLOS = [
    "PARABÉNS! Você ganhou um prêmio de R$ 10.000. Clique aqui agora para resgatar",
    "Oferta imperdível: ganhe dinheiro rápido trabalhando de casa, sem experiência",
    "Seu empréstimo foi pré-aprovado! Dinheiro na conta em 5 minutos, sem consulta ao SPC",
    "Última chance! Desconto de 90% só hoje, compre já e receba um brinde grátis",
    "Você foi selecionado para receber um iPhone grátis. Clique no link e cadastre seus dados",
    "Sua conta bancária foi bloqueada. Atualize seus dados imediatamente neste link",
    "Emagreça 10 kg em 7 dias com este método secreto, resultado garantido",
    "Aumente seu desempenho com nossas pílulas milagrosas, compre sem receita",
    "URGENTE: confirme sua senha agora ou sua conta será cancelada",
    "Ganhe seguidores e curtidas de graça, clique aqui e aproveite a promoção",
    "Investimento com lucro garantido de 300% ao mês, não perca essa oportunidade",
    "Cassino online: bônus de boas-vindas de R$ 500 grátis, jogue agora",
    "Você é o visitante número 1.000.000! Resgate seu cupom premiado clicando aqui",
    "Herança milionária aguardando você. Envie seus dados bancários para receber",
    "Compre barato relógios de marca, 100% originais, frete grátis, clique já",
    "Ganhe dinheiro fácil com criptomoedas, retorno garantido sem risco, cadastre-se",
    "FREE gift card! Click here now to claim your reward, limited time offer",
    "Congratulations, you have won a lottery prize. Send your bank details to claim",
    "Earn money fast from home, no experience needed, act now",
    "Your account has been suspended, verify your password immediately at this link",
    "Cheap meds online without prescription, buy now and save 80 percent",
    "Make $5000 per week guaranteed, click the link and join today",
    "Mensagem exclusiva: limpe seu nome no Serasa hoje mesmo, clique e quite sua dívida com 95% de desconto",
    "Promoção relâmpago!!! Compre 1 leve 5, só hoje, clique agora!!!",
]

HAM_EXEMPLOS = [
    "Olá equipe, segue a ata da reunião de ontem para conferência e aprovação",
    "Bom dia, gostaria de confirmar nossa reunião de alinhamento amanhã às 14h",
    "Anexei o relatório de desempenho da campanha do último trimestre para sua análise",
    "Por favor, revise o briefing do cliente e envie seus comentários até sexta-feira",
    "Lembrete: o prazo para entrega das peças da campanha é na próxima quarta-feira",
    "Obrigado pelo retorno. Vamos agendar uma call para discutir o orçamento do projeto",
    "Segue a proposta comercial conforme conversamos. Fico à disposição para dúvidas",
    "Pessoal, o almoço de confraternização da equipe será na sexta às 12h30",
    "Poderia me enviar a nota fiscal referente ao serviço prestado em outubro?",
    "A apresentação para o cliente está pronta. Podemos revisar juntos hoje à tarde?",
    "Informo que o sistema ficará em manutenção programada neste sábado, das 8h às 10h",
    "Atualizei o cronograma do projeto na pasta compartilhada. Confira as novas datas",
    "Boa tarde, sua solicitação de férias foi aprovada pelo departamento de RH",
    "Gostaria de agendar uma entrevista para a vaga de analista de marketing digital",
    "Segue o contrato assinado. Por favor, confirme o recebimento do documento",
    "Parabéns pelo excelente resultado da campanha, o time está de parabéns",
    "Hi team, please find attached the agenda for tomorrow's planning meeting",
    "Thanks for your feedback on the draft. I will send the revised version by Monday",
    "Could we reschedule our call to Thursday afternoon? I have a conflict today",
    "Attached is the monthly performance report for your review",
    "Reminder: the quarterly review is scheduled for next Tuesday at 10 am",
    "Please confirm your attendance at the client workshop next week",
    "Oi, tudo bem? Conseguiu ver o e-mail que mandei sobre o planejamento de mídia?",
    "Enviei o arquivo com as métricas de engajamento das redes sociais deste mês",
]

TAMANHO_VOCABULARIO = 5000
TAMANHO_SEQUENCIA = 80
SEMENTE = 42


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def construir_modelo(textos: tf.Tensor) -> keras.Model:
    """Cria o modelo com a vetorização de texto embutida."""
    vetorizador = layers.TextVectorization(
        max_tokens=TAMANHO_VOCABULARIO,
        output_mode="int",
        output_sequence_length=TAMANHO_SEQUENCIA,
        standardize="lower_and_strip_punctuation",
    )
    vetorizador.adapt(textos)

    modelo = keras.Sequential(
        [
            keras.Input(shape=(1,), dtype="string"),
            vetorizador,
            layers.Embedding(TAMANHO_VOCABULARIO, 32),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


@st.cache_resource(show_spinner="Treinando o modelo de classificação...")
def treinar_modelo(textos: tuple, rotulos: tuple):
    """Treina o modelo uma única vez e mantém em cache."""
    keras.utils.set_random_seed(SEMENTE)

    x = tf.constant(list(textos))[:, None]
    y = np.array(rotulos, dtype="float32")

    modelo = construir_modelo(tf.constant(list(textos)))
    historico = modelo.fit(x, y, epochs=60, batch_size=8, verbose=0)
    acuracia = float(historico.history["accuracy"][-1])
    return modelo, acuracia


def prever(modelo: keras.Model, mensagem: str) -> float:
    """Retorna a probabilidade (0 a 1) de a mensagem ser spam."""
    entrada = tf.constant([mensagem])[:, None]
    return float(modelo.predict(entrada, verbose=0)[0][0])


# ---------------------------------------------------------------------------
# Dados de treino (padrão ou enviados pelo usuário)
# ---------------------------------------------------------------------------
def carregar_base_padrao():
    textos = SPAM_EXEMPLOS + HAM_EXEMPLOS
    rotulos = [1] * len(SPAM_EXEMPLOS) + [0] * len(HAM_EXEMPLOS)
    return tuple(textos), tuple(rotulos)


def carregar_base_csv(arquivo) -> tuple:
    """Lê um CSV com as colunas 'text' e 'label' (spam/ham ou 1/0)."""
    df = pd.read_csv(io.BytesIO(arquivo.getvalue()))
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("O CSV precisa ter as colunas 'text' e 'label'.")

    mapa = {"spam": 1, "ham": 0, "1": 1, "0": 0, "nao spam": 0, "não spam": 0}
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(mapa)
    df = df.dropna(subset=["text", "label"])
    if df["label"].nunique() < 2 or len(df) < 20:
        raise ValueError(
            "A base deve ter ao menos 20 mensagens, com spam e não spam."
        )
    return tuple(df["text"].astype(str)), tuple(df["label"].astype(int))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("🛡️ Detector de Spam em E-mails")
st.caption(
    "Ferramenta interna de análise de mensagens. Cole o conteúdo do e-mail "
    "abaixo para verificar se ele é spam."
)

with st.sidebar:
    st.header("Configurações")
    limiar = st.slider(
        "Sensibilidade (limiar de spam)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Quanto menor o valor, mais rigorosa é a detecção de spam.",
    )
    st.divider()
    st.subheader("Base de treinamento")
    arquivo_csv = st.file_uploader(
        "Enviar base da empresa (CSV)",
        type="csv",
        help="Colunas obrigatórias: text, label (spam/ham ou 1/0).",
    )

# Seleção da base de dados
textos, rotulos = carregar_base_padrao()
origem_base = "base de demonstração"
if arquivo_csv is not None:
    try:
        textos, rotulos = carregar_base_csv(arquivo_csv)
        origem_base = f"base da empresa ({len(textos)} mensagens)"
    except Exception as erro:  # noqa: BLE001
        st.sidebar.error(f"Não foi possível usar o CSV: {erro}")

modelo, acuracia_treino = treinar_modelo(textos, rotulos)
st.sidebar.caption(
    f"Modelo treinado com a {origem_base}. "
    f"Acurácia no treino: {acuracia_treino:.0%}."
)

mensagem = st.text_area(
    "Mensagem do e-mail",
    height=220,
    placeholder="Cole aqui o assunto e o corpo da mensagem...",
)

if st.button("Analisar mensagem", type="primary", use_container_width=True):
    if not mensagem.strip():
        st.warning("Informe o conteúdo da mensagem para realizar a análise.")
    else:
        with st.spinner("Analisando..."):
            prob_spam = prever(modelo, mensagem.strip())

        eh_spam = prob_spam >= limiar
        st.divider()
        if eh_spam:
            st.error("🚫 **Resultado: a mensagem foi classificada como SPAM.**")
        else:
            st.success("✅ **Resultado: a mensagem NÃO é spam.**")

        col1, col2 = st.columns(2)
        col1.metric("Probabilidade de spam", f"{prob_spam:.1%}")
        col2.metric("Probabilidade de não spam", f"{1 - prob_spam:.1%}")
        st.progress(min(max(prob_spam, 0.0), 1.0))

        st.caption(
            "A classificação é um apoio à decisão. Em caso de dúvida, "
            "encaminhe a mensagem para a equipe de segurança da informação."
        )

st.divider()
st.caption(
    "Aviso: o modelo de demonstração foi treinado com poucos exemplos. "
    "Para uso em produção, envie uma base maior e representativa da empresa."
)
