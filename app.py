import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Configuração da API Key usando st.secrets ---
# Para rodar localmente ou no Streamlit Community Cloud
try:
    # Acesse a chave API via st.secrets
    # No Streamlit Community Cloud, você configurará GEMINI_API_KEY lá.
    # Para testar localmente, você pode criar um arquivo .streamlit/secrets.toml
    # com algo como: GEMINI_API_KEY = "sua_chave_aqui"
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Chave de API do Gemini não encontrada. Por favor, configure 'GEMINI_API_KEY' nos segredos do Streamlit Community Cloud ou em .streamlit/secrets.toml para execução local.")
    st.stop() # Para o aplicativo se a chave não for encontrada

# --- Inicialização do Modelo e das Chains ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    # Prompt Template para texto
    text_prompt_template = """
    Você é um assistente especializado em criar conteúdo didático para o ensino superior.
    Sua tarefa é gerar um {tipo_conteudo} sobre o tema "{tema}" para {nivel_academico}.
    Considere as seguintes instruções adicionais: {detalhes_adicionais}

    Gere um prompt completo e claro para um modelo de linguagem.
    """
    text_prompt = PromptTemplate(
        input_variables=["tipo_conteudo", "tema", "nivel_academico", "detalhes_adicionais"],
        template=text_prompt_template
    )
    chain_text = LLMChain(llm=llm, prompt=text_prompt)

    # Prompt Template para imagem
    image_prompt_template = """
    Crie um prompt detalhado para um gerador de imagens que represente visualmente "{tema_visual}" no estilo "{estilo_arte}".
    A imagem será usada como {aplicacao_didatica}.
    Detalhes específicos para a imagem: {detalhes_imagem}.
    O prompt deve focar na clareza didática e relevância acadêmica.
    """
    image_prompt = PromptTemplate(
        input_variables=["tema_visual", "estilo_arte", "detalhes_imagem", "aplicacao_didatica"],
        template=image_prompt_template
    )
    chain_image = LLMChain(llm=llm, prompt=image_prompt)

except Exception as e:
    st.error(f"Erro na inicialização do modelo ou das chains: {e}")
    st.info("Verifique se a sua chave de API está correta e se você tem acesso ao modelo Gemini.")
    st.stop() # Para o aplicativo se as chains não puderem ser inicializadas

# --- Interface do Usuário com Streamlit ---
st.set_page_config(page_title="Gerador de Prompts Didáticos")
st.title("👨‍🏫 Gerador de Prompts Didáticos para o Ensino Superior 🎓")
st.write("Crie prompts assertivos para modelos de linguagem e geradores de imagem, otimizados para conteúdo acadêmico.")

# --- Seleção do Tipo de Prompt ---
tipo_de_prompt = st.radio(
    "Selecione o tipo de prompt que deseja gerar:",
    ("Prompt para Texto (Modelos de Linguagem)", "Prompt para Imagem (Geradores de Imagem)")
)

st.markdown("---")

# --- Entrada para Prompts de Texto ---
if tipo_de_prompt == "Prompt para Texto (Modelos de Linguagem)":
    st.subheader("📝 Detalhes para o Prompt de Texto")

    tipo_conteudo = st.selectbox(
        "Qual tipo de conteúdo você precisa?",
        ("resumo", "artigo de blog didático", "tópicos para aula", "problema de exercício", "roteiro de vídeo-aula", "explicação de conceito")
    )

    tema = st.text_input("Qual o tema principal do conteúdo? (ex: 'Ciclo de Krebs', 'Revolução Industrial', 'Equações Diferenciais')")

    nivel_academico = st.selectbox(
        "Para qual nível acadêmico é o conteúdo?",
        ("alunos de graduação (introdução)", "alunos de graduação (intermediário)", "alunos de graduação (avançado)", "alunos de pós-graduação")
    )

    detalhes_adicionais = st.text_area(
        "Adicione quaisquer instruções específicas ou requisitos: (ex: 'Deve ter cerca de 300 palavras', 'Focar nas causas e consequências', 'Incluir exemplos práticos')",
        value="Ser conciso e didático.", height=100
    )

    if st.button("Gerar Prompt de Texto"):
        if not tema:
            st.warning("Por favor, preencha o tema.")
        else:
            with st.spinner("Gerando seu prompt de texto..."):
                try:
                    # Chama a chain_text com os inputs do usuário
                    resultado = chain_text.invoke({
                        "tipo_conteudo": tipo_conteudo,
                        "tema": tema,
                        "nivel_academico": nivel_academico,
                        "detalhes_adicionais": detalhes_adicionais
                    })
                    st.subheader("Prompt Gerado para Modelo de Linguagem:")
                    st.code(resultado['text'], language='plaintext')
                    st.download_button(
                        label="Copiar Prompt de Texto",
                        data=resultado['text'],
                        file_name="prompt_texto_gerado.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar o prompt de texto: {e}")
                    st.info("Verifique se sua chave de API está configurada corretamente e se você tem acesso ao Gemini.")

# --- Entrada para Prompts de Imagem ---
elif tipo_de_prompt == "Prompt para Imagem (Geradores de Imagem)":
    st.subheader("🖼️ Detalhes para o Prompt de Imagem")

    tema_visual = st.text_input("O que você quer visualizar? (ex: 'Estrutura de uma célula vegetal', 'Mapa de rotas comerciais do século XV', 'Diagrama de fluxo de um algoritmo')")

    estilo_arte = st.selectbox(
        "Qual estilo artístico você prefere?",
        ("diagrama conceitual", "infográfico", "ilustração científica", "fotorrealista", "pintura acadêmica", "esquemático", "vetorial")
    )

    detalhes_imagem = st.text_area(
        "Detalhes específicos da imagem: (ex: 'foco no núcleo', 'cores neutras', 'perspectiva de cima', 'sem pessoas')",
        value="Ser claro, com boa iluminação e sem texto.", height=100
    )

    aplicacao_didatica = st.text_input(
        "Onde a imagem será usada? (ex: 'slide de apresentação', 'material de apoio impresso', 'livro didático digital')",
        value="slide de apresentação em sala de aula"
    )

    if st.button("Gerar Prompt de Imagem"):
        if not tema_visual:
            st.warning("Por favor, preencha o tema visual.")
        else:
            with st.spinner("Gerando seu prompt de imagem..."):
                try:
                    # Chama a chain_image com os inputs do usuário
                    resultado = chain_image.invoke({
                        "tema_visual": tema_visual,
                        "estilo_arte": estilo_arte,
                        "detalhes_imagem": detalhes_imagem,
                        "aplicacao_didatica": aplicacao_didatica
                    })
                    st.subheader("Prompt Gerado para Gerador de Imagem:")
                    st.code(resultado['text'], language='plaintext')
                    st.download_button(
                        label="Copiar Prompt de Imagem",
                        data=resultado['text'],
                        file_name="prompt_imagem_gerado.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Erro ao gerar o prompt de imagem: {e}")
                    st.info("Verifique se sua chave de API está configurada corretamente e se você tem acesso ao Gemini.")