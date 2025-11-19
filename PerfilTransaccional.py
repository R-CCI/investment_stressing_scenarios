import streamlit as st
import tempfile
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import json
import duckdb
import asyncio
import nest_asyncio
from datetime import datetime, timedelta
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
import sys
import io
import os
from PIL import Image
import base64
from rich.prompt import Prompt
from langchain_community.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.schema import HumanMessage
import requests
import time
import random
import ast
import re
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from reportlab.lib.pagesizes import letter
import ssl, certifi
ssl._create_default_https_context = ssl._create_unverified_context
from reportlab.pdfgen import canvas
import textwrap
from fpdf import FPDF
from PIL import Image
#import pytesseract  # for OCR on images
import pdfplumber
import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Streamlit UI
st.set_page_config(page_title="Compliance Financial Analyzer", layout="wide")
st.title("Perfil Transaccional")
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pdfplumber

class DetectorAgent:
    """Detect PDF type using embeddings and similarity search."""

    @staticmethod
    def detect_pdf_type(pdf_path: str) -> str:
        try:
            # 1️⃣ Embed the labels
            labels = ["TransUnion Report", "Estado de Cuenta"]
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # picks up API key from env
            label_vectors = [embeddings.embed_query(label) for label in labels]

            # 2️⃣ Create FAISS index
            db = FAISS.from_embeddings(label_vectors, labels)

            # 3️⃣ Read PDF text (first page)
            with pdfplumber.open(pdf_path) as pdf:
                text = " ".join([page.extract_text() for page in pdf.pages[:1] if page.extract_text()])

            if not text.strip():
                return "Unknown"

            # 4️⃣ Embed PDF text
            pdf_vector = embeddings.embed_query(text)

            # 5️⃣ Query FAISS
            result = db.similarity_search_by_vector(pdf_vector, k=1)
            return result[0].page_content  # returns closest label

        except Exception as e:
            return f"Error reading PDF: {e}"


class SupervisorAgent:
    """
    Summarizes or filters the ComplianceAgent's analysis to output only the numeric range.
    """

    @staticmethod
    def extract_range(analysis_text: str):
        """Extracts income range from ComplianceAgent output"""
        llm = ChatOpenAI(model="gpt-5", temperature=1)

        prompt_text = """
        Eres un supervisor analítico. 
        Tu tarea es leer el siguiente texto, que es un análisis financiero completo, 
        y devolver **únicamente** el rango de ingresos mensuales estimados en formato:

        LowerLimit - UpperLimit

        Ejemplo de salida válida:
        45,000 - 65,000

        Texto del análisis:
        {analysis_text}
        """

        template = PromptTemplate(
            input_variables=["analysis_text"], 
            template=prompt_text
        )
        chain = LLMChain(llm=llm, prompt=template)

        result = chain.run({"analysis_text": analysis_text})
        return result.strip()
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
      
class ComplianceAgent:
    """Analyze TransUnion PDF reports using structured financial tables and LLM analysis"""

    @staticmethod
    def extract_financial_data(pdf_path, usd_to_dop=60):
        """Extract financial data table from PDF and preprocess"""
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]  # assuming table is on the first page
            table = page.extract_table()

        if table is None:
            raise ValueError("No table found in PDF page")

        header_row1 = table[1]
        header_row2 = table[2]
        data_rows = table[3:]

        # Combine headers
        column_names = []
        prev_col = ''
        for i, col in enumerate(header_row1):
            if col is not None:
                prev_col = col
            sub_col = header_row2[i]
            if sub_col is not None:
                column_names.append(f"{prev_col} ({sub_col})")
            else:
                column_names.append(col)

        df = pd.DataFrame(data_rows, columns=column_names)
        df.columns = [col.replace('\n', ' ') for col in df.columns]

        # Columns to convert to numeric
        numeric_cols = [
            'CANT. CTAS',
            'MONTO DE CREDITO (RD$)',
            'MONTO DE CREDITO (US$)',
            'BALANCE ACTUAL (RD$)',
            'BALANCE ACTUAL (US$)',
            'ATRASO ACTUAL (RD$)',
            'ATRASO ACTUAL (US$)'
        ]
        percent_cols = [
            '% UTILIZACION (RD$)',
            '% UTILIZACION (US$)'
        ]

        # Clean numeric columns
        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(',', '').replace('', '0').astype(np.int64)
        for col in percent_cols:
            df[col] = df[col].astype(str).str.replace('%', '').replace('', '0').astype(np.int64)

        # Keep only subtotal rows
        df = df[df['SUSCRIPTOR'].str.contains('SUB-TOTAL', na=False)]

        # Compute consolidated financial columns
        df['Total Credit Amount'] = df['MONTO DE CREDITO (RD$)'] + df['MONTO DE CREDITO (US$)'] * usd_to_dop
        df['Amount Remaining to Pay'] = df['BALANCE ACTUAL (RD$)'] + df['BALANCE ACTUAL (US$)'] * usd_to_dop
        df['Financial Arrears'] = df['ATRASO ACTUAL (RD$)'] + df['ATRASO ACTUAL (US$)'] * usd_to_dop

        df = df[['TIPO CTA', 'Total Credit Amount', 'Amount Remaining to Pay', 'Financial Arrears']]
        df['TIPO CTA'] = df['TIPO CTA'].map({'TC': 'Credit Card', 'TEL': 'Telecommunications', 'PR': 'Loans'})

        loader = PyPDFLoader(pdf_path)
        pages = []
        for page in loader.lazy_load():
          pages.append(page)
        score_transunion = pages[0].page_content[pages[0].page_content.find("TRANSUNION CREDITVISION SCORE\nPuntuación\n")+len("TRANSUNION CREDITVISION SCORE\nPuntuación\n"):pages[0].page_content.find("TRANSUNION CREDITVISION SCORE\nPuntuación\n")+len("TRANSUNION CREDITVISION SCORE\nPuntuación\n")+3]

        return df, int(score_transunion)

    @staticmethod
    def investment_limit(pdf_path, usd_to_dop=60):
        """Run analysis pipeline"""
        try:
            # Step 1: Extract structured financial data
            df, score_transunion = ComplianceAgent.extract_financial_data(pdf_path, usd_to_dop=usd_to_dop)

            # Step 2: Convert DataFrame to JSON
            df_json = df.to_json(orient="records")

            # Step 3: LLM analysis
            llm = ChatOpenAI(model="gpt-5", temperature=1)
            prompt_text = """
            Eres un analista financiero experimentado especializado en el mercado de la República Dominicana.

            Analiza los siguientes datos financieros estructurados (JSON) sobre las deudas de una persona. Nuestro objetivo es estimar el ingreso de la persona
            y luego el diferencial neto que le queda luego de sus compromisos, que podemos llamarle su capacidad de ahorro.
            Considera que los prestamos pueden ser consumo, vehiculo o hipotecario, puedes hacer varios escenarios.
            Adicionalmente, ten en cuenta que en RD la tasa de interes anual de TC es de 60%, pero esto solo se aplica si te atrasas. Si tu tarjeta esta al dia, no
            pagas intereses. Adicionalmente, analiza bien los demographics de RC, porque la gente no gana bien y a veces no destina el 30% o 40% de sus ingresos a deuda,
            sino una fraccion mayor o la totalidad incluso.
            {df_json} y el score de Transunion es {score_transunion}

            Instrucciones:
            1. Suma la deuda total en RD$.
            2. Suma los pagos mensuales.
            3. Calcula el ingreso bruto mensual utilizando ratios DTI (30-40% con saldos limpios, 45-60% con morosidad).
            4. Indica el rango de ingresos con un nivel de confianza.
            5. Proporciona un análisis cualitativo: historial de pagos, composición de la deuda, progreso del préstamo, perfil de riesgo.

            Presentar un análisis profesional claro.
            """

            mm_template = PromptTemplate(input_variables=["df_json", "score_transunion"], template=prompt_text)
            query_chain = LLMChain(llm=llm, prompt=mm_template)
            result = query_chain.run({"df_json": df_json, "score_transunion":score_transunion})

            return result

        except Exception as e:
            return f"Error due to {e}"
    @staticmethod
    def investment_limit_embebbings(db_e, usd_to_dop=60):
        """Run analysis pipeline"""
        try:
            query = "Estimate monthly income range based on the financial data"
            retrieved_docs = db_e.similarity_search(query, k=3)  # returns list of Documents

            # Combine the retrieved content into a single context string
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            if "transunion" in (context.lower()):
                            prompt_text = """
            Eres un analista financiero experimentado especializado en el mercado de la República Dominicana.

            Analiza los siguientes embbebings sobre las deudas de una persona. Nuestro objetivo es estimar el ingreso de la persona
            y luego el diferencial neto que le queda luego de sus compromisos, que podemos llamarle su capacidad de ahorro.
            Considera que los prestamos pueden ser consumo, vehiculo o hipotecario, puedes hacer varios escenarios.
            Adicionalmente, ten en cuenta que en RD la tasa de interes anual de TC es de 60%, pero esto solo se aplica si te atrasas. Si tu tarjeta esta al dia, no
            pagas intereses. Adicionalmente, analiza bien los demographics de RC, porque la gente no gana bien y a veces no destina el 30% o 40% de sus ingresos a deuda,
            sino una fraccion mayor o la totalidad incluso.
            {context} y el score de Transunion.

            Instrucciones:
            1. Suma la deuda total en RD$.
            2. Suma los pagos mensuales.
            3. Calcula el ingreso bruto mensual utilizando ratios DTI (30-40% con saldos limpios, 45-60% con morosidad).
            4. Indica el rango de ingresos con un nivel de confianza.
            5. Proporciona un análisis cualitativo: historial de pagos, composición de la deuda, progreso del préstamo, perfil de riesgo.

            Presentar un análisis profesional claro.
            """
            else:
                prompt_text = """
            Eres un analista financiero experimentado especializado en el mercado de la República Dominicana.
            Se te suministra un estado de cuenta con fechas y movimientos del clientes. Nuestro objetivo es estimar el ingreso de la persona
            y luego el diferencial neto que le queda luego de sus compromisos, que podemos llamarle su capacidad de ahorro.
            Considera que el cliente tiene ingresos y egresos en el estado de cuenta con saldos positivos y negativos.
            Adicionalmente, ten en cuenta la fecha, por ejemplo si se te suministra 2 meses de informacion, deberemos ajustarlo para buscar la equivalencia de ingresos en un solo mes. 
            Igualmente si se te da menos de un mes de informacion. Aqui esta la data {context}

            Instrucciones:
            1. Suma las entradas de dinero, principalmente relacionadas con nomina o pagos de cliente por si es independiente.
            2. Suma los gastos.
            3. Valida el periodo en que suceden y calcula el ingreso bruto mensual utilizando ratios DTI (30-40% con saldos limpios, 45-60% con morosidad).
            4. Indica el rango de ingresos con un nivel de confianza.
            5. Proporciona un análisis cualitativo.

            Presentar un análisis profesional claro.
            """
            # Step 3: LLM analysis
            llm = ChatOpenAI(model="gpt-5", temperature=1)


            mm_template = PromptTemplate(input_variables=["context"], template=prompt_text)
            query_chain = LLMChain(llm=llm, prompt=mm_template)
            result = query_chain.run({"context": context})

            return result

        except Exception as e:
            return f"Error due to {e}"
    @staticmethod
    def investment_limit_embebbings_estado_cuenta(db_e, usd_to_dop=60):
        """Run analysis pipeline"""
        try:
            query = "Estimate monthly income range based on the financial data"
            retrieved_docs = db_e.similarity_search(query, k=3)  # returns list of Documents

            # Combine the retrieved content into a single context string
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            # Step 3: LLM analysis
            llm = ChatOpenAI(model="gpt-5", temperature=1)
            prompt_text = """
            Eres un analista financiero experimentado especializado en el mercado de la República Dominicana.
            Se te suministra un estado de cuenta con fechas y movimientos del clientes. Nuestro objetivo es estimar el ingreso de la persona
            y luego el diferencial neto que le queda luego de sus compromisos, que podemos llamarle su capacidad de ahorro.
            Considera que el cliente tiene ingresos y egresos en el estado de cuenta con saldos positivos y negativos.
            Adicionalmente, ten en cuenta la fecha, por ejemplo si se te suministra 2 meses de informacion, deberemos ajustarlo para buscar la equivalencia de ingresos en un solo mes. 
            Igualmente si se te da menos de un mes de informacion. Aqui esta la data {context}

            Instrucciones:
            1. Suma las entradas de dinero, principalmente relacionadas con nomina o pagos de cliente por si es independiente.
            2. Suma los gastos.
            3. Valida el periodo en que suceden y calcula el ingreso bruto mensual utilizando ratios DTI (30-40% con saldos limpios, 45-60% con morosidad).
            4. Indica el rango de ingresos con un nivel de confianza.
            5. Proporciona un análisis cualitativo.

            Presentar un análisis profesional claro.
            """

            mm_template = PromptTemplate(input_variables=["context"], template=prompt_text)
            query_chain = LLMChain(llm=llm, prompt=mm_template)
            result = query_chain.run({"context": context})

            return result

        except Exception as e:
            return f"Error due to {e}"


uploaded_file = st.file_uploader("Suba un archivo", type=["pdf", "png", "jpg", "jpeg"])



if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        st.info("Procesando el archivo ...")
        # Try to extract structured data
        df, score = ComplianceAgent.extract_financial_data(pdf_path)
        st.success(f"✅ Data —  Score: {score}")
        st.dataframe(df)

        # Run the LLM analysis
        st.subheader("📊 LLM Límite de Inversión")
        result = ComplianceAgent.investment_limit(pdf_path)
        income_range = SupervisorAgent.extract_range(result)
        st.success(f"#### Ingresos estimados: **{income_range} DOP**")

        st.subheader("📊 LLM Análisis")
        st.write(result)

    except Exception as e:
        st.warning(f"⚠️ Extracción de data estructurada falló: {e}")
        st.write("Intentando analizar mediante embeddings...")

        # Fallback: embedding-based analysis
        try:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                text_content = "\n".join([p.page_content for p in pages])
                st.write("✅ Archivos leídos.")

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                db = FAISS.from_texts([text_content], embedding=embeddings)
                
                st.write("✅ Embeddings creados.")
                #st.code(text_content[:1500] + "..." if len(text_content) > 1500 else text_content)
                #doc_type = DetectorAgent().detect_pdf_type(pdf_path)
                #st.write(doc_type)
                # if doc_type == "TransUnion Report":
                #     result = ComplianceAgent.investment_limit_embebbings(db)
                # else:
                result = ComplianceAgent.investment_limit_embebbings(db)
                income_range = SupervisorAgent.extract_range(result)
                st.success(f"#### Ingresos estimados: **{income_range} DOP**")
        
                st.subheader("📊 LLM Análisis")
                st.markdown(f"""
                <style>
                .font-regular {{
                    font-family: 'Roboto', sans-serif;
                    font-size: 15px;
                }}
                .font-italic {{
                    font-family: 'Georgia', serif;
                    font-style: italic;
                    font-size: 16px;
                }}
                </style>
                
                <body>
                <div class="font-regular">
                <p> {result}</p>
                </div>
                </body>""", unsafe_allow_html=True)
            elif uploaded_file.type.startswith("image"):
                save_path = os.path.join(uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image = Image.open(save_path).convert("RGB")
                base64_image = encode_image(save_path)
                # vector = model.encode([image])
                # vectors = np.array([vector], dtype=np.float32)
                #st.image(image, caption="Uploaded Image", use_column_width=True)
                #text_content = pytesseract.image_to_string(image)
                st.write("✅ Archivos leídos.")
    
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                texts = ["financial_doc.png"]  # must be same length as vectors
                
                # Create FAISS index
                db = FAISS.from_texts([base64_image], embedding=embeddings)
                
                st.write("✅ Embeddings creados.")
                #st.code(text_content[:1500] + "..." if len(text_content) > 1500 else text_content)
                result = ComplianceAgent.investment_limit_embebbings(db)
                api_key = st.secrets["OPENAI_API_KEY"]
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                payload = {
                    "model": "gpt-5",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": """Eres un analista financiero experimentado especializado en el mercado de la República Dominicana.
                                Se te suministra un estado de cuenta con fechas y movimientos del clientes. Nuestro objetivo es estimar el ingreso de la persona
                                y luego el diferencial neto que le queda luego de sus compromisos, que podemos llamarle su capacidad de ahorro.
                                Considera que el cliente tiene ingresos y egresos en el estado de cuenta con saldos positivos y negativos.
                                Adicionalmente, ten en cuenta la fecha, por ejemplo si se te suministra 2 meses de informacion, deberemos ajustarlo para buscar la equivalencia de ingresos en un solo mes. 
                                Igualmente si se te da menos de un mes de informacion.
                    
                                Instrucciones:
                                1. Suma la deuda total en RD$.
                                2. Suma los pagos mensuales.
                                3. Calcula el ingreso bruto mensual utilizando ratios DTI (30-40% con saldos limpios, 45-60% con morosidad).
                                4. Indica el rango de ingresos con un nivel de confianza.
                                5. Proporciona un análisis cualitativo: historial de pagos, composición de la deuda, progreso del préstamo, perfil de riesgo.
                    
                                Presentar un análisis profesional claro.
                                """},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                }
                
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                description = response.json()['choices'][0]['message']['content']
        
                st.subheader("📊 LLM Análisis")
                income_range = SupervisorAgent.extract_range(description)
                st.success(f"#### Ingresos estimados: **{income_range} DOP**")
                st.markdown(f"""
                <style>
                .font-regular {{
                    font-family: 'Roboto', sans-serif;
                    font-size: 15px;
                }}
                .font-italic {{
                    font-family: 'Georgia', serif;
                    font-style: italic;
                    font-size: 16px;
                }}
                </style>
                <body>
                <div class="font-regular">
                <p> {description}</p>
                </div>
                </body>""", unsafe_allow_html=True)

        except Exception as e2:
            st.error(f"❌ Error al crear embeddings: {e2}")























































