import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import toml
from openai import OpenAI

st.set_page_config(page_title="HW - Versão A (Abas)", layout="wide")
st.title("Calculadora Genética Completa — Versão A (Abas)")
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
secrets = toml.load("/etc/secrets/secrets.toml")
client = OpenAI(api_key=secrets["OPENAI_API_KEY"])

tabs = st.tabs(["Hardy–Weinberg", "Simulação Evolutiva", "Testes/Questões", "Agente IA"])
# --------------------------------
# ABA 1: HARDY–WEINBERG
# --------------------------------
with tabs[0]:
   st.header("Calculadora Hardy–Weinberg")
   mode = st.radio("Entrada disponível:", ("AA, Aa, aa", "Só aa conhecido (fenótipo recessivo)", "Frequências/percentuais"))
   if mode == "AA, Aa, aa":
       N_AA = st.number_input("Número observados AA:", min_value=0, value=36)
       N_Aa = st.number_input("Número observados Aa:", min_value=0, value=48)
       N_aa = st.number_input("Número observados aa:", min_value=0, value=16)
       N = N_AA + N_Aa + N_aa
       p = (2*N_AA + N_Aa) / (2*N) if N>0 else 0
       q = 1 - p
       p2 = p**2
       pq2 = 2*p*q
       q2 = q**2
       E_AA = p2 * N
       E_Aa = pq2 * N
       E_aa = q2 * N
       st.subheader("Frequências Alélicas")
       st.write(f"p = {p:.4f}    q = {q:.4f}")
       st.subheader("Frequências Genotípicas (esperadas)")
       df = pd.DataFrame({
           "Genótipo": ["AA","Aa","aa"],
           "Freq (teórica)": [p2, pq2, q2],
           "Esperado (n)":[E_AA, E_Aa, E_aa],
           "Observado (n)":[N_AA, N_Aa, N_aa]
       })
       st.dataframe(df.round(4))
       chi = 0.0
       for O,E in zip([N_AA,N_Aa,N_aa], [E_AA,E_Aa,E_aa]):
           if E>0:
               chi += (O-E)**2 / E
       st.subheader("Teste Qui-Quadrado")
       st.write(f"χ² = {chi:.4f}  (df = 1, critério α=0.05 → 3.84)")
       if chi < 3.84:
           st.success("Diferença NÃO significativa — População APARENTA em equilíbrio.")
       else:
           st.error("Diferença significativa — População POSSIVELMENTE fora do equilíbrio.")
   elif mode == "Só aa conhecido (fenótipo recessivo)":
       N = st.number_input("Tamanho da população (N):", min_value=1, value=10000)
       n_aa = st.number_input("Número de indivíduos com fenótipo recessivo (aa):", min_value=0, value=900)
       q2 = n_aa / N
       q = math.sqrt(q2)
       p = 1 - q
       p2 = p*p
       pq2 = 2*p*q
       n_AA = p2 * N
       n_Aa = pq2 * N
       st.write(f"q² = {q2:.4f} → q = {q:.4f} → p = {p:.4f}")
       st.write(f"2pq = {pq2:.4f} → heterozigotos esperados (n) = {n_Aa:.0f}")
   else:
       choice = st.selectbox("O que você tem?", ["% dominante (fenótipo)", "% recessivo (fenótipo)", "q (frequência)"])
       val = st.number_input("Valor (0-1 ou 0-100):", value=64.0)
       if val>1: val = val/100
       if choice == "% dominante (fenótipo)":
           rec = 1 - val
           q = math.sqrt(rec)
           p = 1-q
       elif choice == "% recessivo (fenótipo)":
           q = math.sqrt(val)
           p = 1-q
       else:
           q = val
           p = 1-q
       st.write(f"p = {p:.4f}, q = {q:.4f}")
       st.write(f"p² = {p**2:.4f}, 2pq = {(2*p*q):.4f}, q² = {q**2:.4f}")
# --------------------------------
# ABA 2: SIMULAÇÃO EVOLUTIVA
# --------------------------------
with tabs[1]:
   st.header("Simulação Evolutiva (seleção/mutação/migração)")
   N = st.number_input("População total (N):", min_value=2, value=1000)
   q0 = st.number_input("q inicial (0-1):", min_value=0.0, max_value=1.0, value=0.3)
   gens = st.number_input("Gerações:", min_value=1, value=50)
   sel = st.slider("Seleção contra aa (fração eliminada por geração):", 0.0, 1.0, 0.0)
   mu = st.number_input("Taxa de mutação (A→a):", min_value=0.0, value=0.0)
   mig = st.number_input("Migração (fração imigrante):", min_value=0.0, value=0.0)
   if mig>0:
       q_im = st.number_input("q dos imigrantes:", min_value=0.0, max_value=1.0, value=0.5)
   else:
       q_im = None
   rows = []
   q = q0
   for g in range(gens):
       p = 1-q
       p2, pq2, q2 = p*p, 2*p*q, q*q
       q2_sel = q2 * (1-sel)
       total = p2 + pq2 + q2_sel
       p_post = (p2 + pq2/2) / total
       q_post = 1 - p_post
       q_mut = q_post + mu*p_post - mu*q_post
       p_mut = 1 - q_mut
       if mig>0:
           q_new = (1-mig)*q_mut + mig*q_im
       else:
           q_new = q_mut
       rows.append([g, q, p, p*p, 2*p*q, q*q])
       q = float(max(0.0, min(1.0, q_new)))
   df_sim = pd.DataFrame(rows, columns=["Geração","q","p","p²","2pq","q²"])
   st.dataframe(df_sim.round(4))
   st.line_chart(df_sim.set_index("Geração")[["p","q"]])
# --------------------------------
# ABA 3: TESTES / QUESTÕES
# --------------------------------
with tabs[2]:
   st.header("Responder Questões de Múltipla Escolha")
   N_q = st.number_input("População (N):", min_value=1, value=10000)
   n_aa_q = st.number_input("Número de aa:", min_value=0, value=900)
   opts = st.text_area("Opções:", value="a) 1800\nb)1300\nc) 4200\nd) 3000\ne) 3020")
   if st.button("Resolver Questão"):
       q2 = n_aa_q / N_q
       q_val = math.sqrt(q2)
       p_val = 1 - q_val
       hetero_frac = 2*p_val*q_val
       hetero_n = hetero_frac * N_q
       st.write(f"Heterozigotos esperados = {hetero_n:.0f}")
       import re
       chosen = None
       for line in opts.splitlines():
           m = re.search(r"([0-9]+)", line.replace(",",""))
           if m:
               if int(m.group(1)) == round(hetero_n):
                   chosen = line
       if chosen:
           st.success(f"Alternativa correta: {chosen}")
       else:
           st.warning("Nenhuma alternativa corresponde ao valor encontrado.")
# --------------------------------
# ABA 4: AGENTE IA
# --------------------------------
with tabs[3]:
   st.header("Agente IA — Resolve qualquer questão de genética")
   pergunta = st.text_area("Escreva sua questão:")
   if st.button("Perguntar à IA"):
       if pergunta.strip() == "":
           st.error("Digite uma pergunta.")
       else:
           prompt_sistema = """
           Você é um especialista em genética e Hardy–Weinberg.
           Tarefas:
           - Resolver qualquer exercício de genética.
           - Calcular p, q, p², 2pq, q².
           - Determinar contagens esperadas.
           - Determinar número de heterozigotos.
           - Fazer qui-quadrado quando houver dados.
           - Explicar passo a passo.
           """
           response = client.chat.completions.create(
               model="gpt-4.1-mini",
               messages=[
                   {"role": "system", "content": prompt_sistema},
                   {"role": "user", "content": pergunta}
               ]
           )
           st.write(response.choices[0].message.content)