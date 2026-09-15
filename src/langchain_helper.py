import os
import sys
import json
import pdfplumber
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.logger import logging
from src.exception import CustomException

load_dotenv()


# ─── LLM Setup ───────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY"),
    temperature=0.6
)


# ─── Function 1: PDF Sample Test Extract ──────────────────────────
def extract_text_from_file(pdf_file) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + '\n'

        logging.info('PDF extraction completed successfully.')
        return text

    except Exception as e:
        raise CustomException(e, sys)



# ─── Function 2: Gemini will Parse Data ────────────────────────────
def extract_data_from_pdf(pdf_file) -> dict:
    try:
        pdf_text = extract_text_from_file(pdf_file)

        prompt = PromptTemplate(
            input_variables=["pdf_text"],
            template="""
            You are a medical data extraction assistant.

            Extract the following fields from the health report text and return ONLY a valid JSON object.
            Do not include any explanation, markdown, or extra text — only raw JSON.

            Fields to extract:
            - state (string: one of Assam, Manipur, Meghalaya, Mizoram, Nagaland, Arunachal Pradesh, Sikkim, Tripura)
            - age (integer)
            - season (string: Winter, Summer, Monsoon, Post-Monsoon)
            - month (integer: 1-12)
            - flooding (integer: 0 or 1)
            - water_source (string: River, Pond, Open Well, Rainwater, Tanker, Borewell, Piped)
            - water_treatment (string: Untreated, Boiled, Filtered, Chlorinated)
            - handwashing_practice (string: Never, Sometimes, Always)
            - toilet_access (integer: 0 or 1)
            - open_defecation_rate (float)
            - sewage_treatment_pct (float)
            - water_quality_index (float)
            - ph (float)
            - turbidity_ntu (float)
            - dissolved_oxygen_mg_l (float)
            - bod_mg_l (float)
            - fecal_coliform_per_100ml (integer)
            - total_coliform_per_100ml (integer)
            - tds_mg_l (float)
            - nitrate_mg_l (float)
            - fluoride_mg_l (float)
            - arsenic_ug_l (float)
            - avg_temperature_c (float)
            - avg_rainfall_mm (float)
            - avg_humidity_pct (float)
            - symptom_diarrhea (integer: 0 or 1)
            - symptom_vomiting (integer: 0 or 1)
            - symptom_fever (integer: 0 or 1)
            - symptom_abdominal_pain (integer: 0 or 1)
            - symptom_dehydration (integer: 0 or 1)
            - symptom_jaundice (integer: 0 or 1)
            - symptom_bloody_stool (integer: 0 or 1)
            - symptom_skin_rash (integer: 0 or 1)

            Health Report Text:
            {pdf_text}

            Return only JSON:
            """
        )

        chain  = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(pdf_text=pdf_text)

        # Clean response
        result = result.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        result = result.strip()

        data = json.loads(result)
        logging.info("Data extracted from PDF successfully")
        return data

    except Exception as e:
        raise CustomException(e, sys)

# ─── State Helpline Numbers ───────────────────────────────────────
STATE_HELPLINES = {
    'Assam': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '0361-2237240',
        'govt_health': 'Directorate of Health Services, Assam — 0361-2237006'
    },
    'Manipur': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '0385-2450137',
        'govt_health': 'Directorate of Health Services, Manipur — 0385-2411447'
    },
    'Meghalaya': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '0364-2224318',
        'govt_health': 'Directorate of Health Services, Meghalaya — 0364-2220458'
    },
    'Mizoram': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '0389-2322694',
        'govt_health': 'Directorate of Health Services, Mizoram — 0389-2325584'
    },
    'Nagaland': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '0370-2271697',
        'govt_health': 'Directorate of Health Services, Nagaland — 0370-2291782'
    },
    'Arunachal Pradesh': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '0360-2212624',
        'govt_health': 'Directorate of Health Services, Arunachal Pradesh — 0360-2212056'
    },
    'Sikkim': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '03592-202323',
        'govt_health': 'Directorate of Health Services, Sikkim — 03592-202439'
    },
    'Tripura': {
        'health_helpline': '104',
        'emergency': '108',
        'disease_control': '0381-2415583',
        'govt_health': 'Directorate of Health Services, Tripura — 0381-2226862'
    }
}

# ─── Disease Specialist Map ───────────────────────────────────────
DISEASE_SPECIALIST = {
    'Cholera':       'Gastroenterologist / Infectious Disease Specialist',
    'Typhoid':       'General Physician / Infectious Disease Specialist',
    'Dysentery':     'Gastroenterologist',
    'Hepatitis_A':   'Hepatologist / Gastroenterologist',
    'Hepatitis_E':   'Hepatologist',
    'Leptospirosis': 'Infectious Disease Specialist',
    'Giardiasis':    'General Physician',
    'No_Disease':    None
}


# ─── Function 3: Health Advisory Generate karo ───────────────────