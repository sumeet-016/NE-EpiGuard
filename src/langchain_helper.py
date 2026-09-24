import os
import sys
import json
import pdfplumber
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from src.logger import logging
from src.exception import CustomException

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

STATE_HELPLINES = {
    'Assam':              {'disease_control': '0361-2237240', 'govt_health': 'DHS Assam — 0361-2237006'},
    'Manipur':            {'disease_control': '0385-2450137', 'govt_health': 'DHS Manipur — 0385-2411447'},
    'Meghalaya':          {'disease_control': '0364-2224318', 'govt_health': 'DHS Meghalaya — 0364-2220458'},
    'Mizoram':            {'disease_control': '0389-2322694', 'govt_health': 'DHS Mizoram — 0389-2325584'},
    'Nagaland':           {'disease_control': '0370-2271697', 'govt_health': 'DHS Nagaland — 0370-2291782'},
    'Arunachal Pradesh':  {'disease_control': '0360-2212624', 'govt_health': 'DHS Arunachal — 0360-2212056'},
    'Sikkim':             {'disease_control': '03592-202323', 'govt_health': 'DHS Sikkim — 03592-202439'},
    'Tripura':            {'disease_control': '0381-2415583', 'govt_health': 'DHS Tripura — 0381-2226862'},
}

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


def extract_text_from_pdf(pdf_file) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        logging.info("PDF text extracted successfully")
        return text
    except Exception as e:
        raise CustomException(e, sys)


def extract_data_from_pdf(pdf_file) -> dict:
    try:
        pdf_text = extract_text_from_pdf(pdf_file)
        time.sleep(2)

        prompt = PromptTemplate(
            input_variables=["pdf_text"],
            template="""
You are a medical data extraction assistant.
Extract the following fields from the health report and return ONLY valid JSON.
No explanation, no markdown, no extra text — only raw JSON.

Fields:
state, age, season, month, flooding, water_source, water_treatment,
handwashing_practice, toilet_access, open_defecation_rate, sewage_treatment_pct,
water_quality_index, ph, turbidity_ntu, dissolved_oxygen_mg_l, bod_mg_l,
fecal_coliform_per_100ml, total_coliform_per_100ml, tds_mg_l, nitrate_mg_l,
fluoride_mg_l, arsenic_ug_l, avg_temperature_c, avg_rainfall_mm, avg_humidity_pct,
symptom_diarrhea, symptom_vomiting, symptom_fever, symptom_abdominal_pain,
symptom_dehydration, symptom_jaundice, symptom_bloody_stool, symptom_skin_rash

Health Report:
{pdf_text}

Return only JSON:
"""
        )

        chain  = prompt | llm
        result = chain.invoke({"pdf_text": pdf_text})
        result = result.content.strip()

        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]


        if hasattr(result, 'content'): result = result.content.strip()
        elif isinstance(result, list): result = result[0].content.strip() if result else ""
        else: result = str(result).strip()

        data = json.loads(result)
        logging.info("Data extracted from PDF successfully")
        return data

    except Exception as e:
        raise CustomException(e, sys)


def generate_health_advisory(disease: str, probability: float, patient_data: dict) -> str:
    try:
        symptom_cols  = ['symptom_diarrhea', 'symptom_vomiting', 'symptom_fever',
                         'symptom_abdominal_pain', 'symptom_dehydration', 'symptom_jaundice',
                         'symptom_bloody_stool', 'symptom_skin_rash']
        symptom_names = ['Diarrhea', 'Vomiting', 'Fever', 'Abdominal Pain',
                         'Dehydration', 'Jaundice', 'Bloody Stool', 'Skin Rash']
        present_symptoms = [symptom_names[i] for i, col in enumerate(symptom_cols)
                            if patient_data.get(col, 0) == 1]
        symptoms_str = ", ".join(present_symptoms) if present_symptoms else "None"

        state     = patient_data.get('state', 'Assam')
        helplines = STATE_HELPLINES.get(state, STATE_HELPLINES['Assam'])
        specialist = DISEASE_SPECIALIST.get(disease, 'General Physician')

        helpline_section = ""
        if probability >= 70.0 and disease != 'No_Disease':
            helpline_section = f"""

7. HELPLINE AND DOCTOR RECOMMENDATION (Confidence >= 70%)
   ** High confidence — immediate medical consultation recommended **

   Recommended Specialist : {specialist}

   State Helplines ({state}):
   - National Health Helpline : 104 (free, 24x7)
   - Emergency Ambulance      : 108 (free, 24x7)
   - Disease Control Officer  : {helplines['disease_control']}
   - State Health Department  : {helplines['govt_health']}
   - NCDC National Helpline   : 011-23921401

   Please call 104 or 108 immediately if symptoms are severe.
"""

        prompt = PromptTemplate(
            input_variables=["disease", "probability", "state", "water_source",
                             "water_treatment", "handwashing", "flooding",
                             "symptoms", "helpline_section"],
            template="""
You are a public health advisor following WHO guidelines for waterborne disease 
management in Northeast India.

Patient Assessment:
- Predicted Disease   : {disease}
- Confidence          : {probability}%
- Location            : {state}, Northeast India
- Water Source        : {water_source}
- Water Treatment     : {water_treatment}
- Handwashing Practice: {handwashing}
- Flooding Event      : {flooding}
- Symptoms Present    : {symptoms}

Provide structured advisory:

1. DISEASE OVERVIEW
   (WHO definition, transmission, burden in South Asia — 2-3 lines)

2. IMMEDIATE ACTIONS (WHO Emergency Protocol)
   - 3-4 steps to take NOW
   - Include WHO ORS recommendation if applicable

3. WATER SAFETY TIPS (WHO WASH Guidelines)
   - 3 tips specific to water source: {water_source}

4. PREVENTION (WHO Protocol)
   - 3-4 WHO recommended measures
   - Include WHO 5 moments of hand hygiene

5. WARNING SIGNS (WHO Red Flags)
   - Symptoms needing immediate hospitalization
   - High risk groups: children under 5, elderly, pregnant women

6. COMMUNITY ACTION (WHO One Health)
   - 2 community level outbreak prevention actions

{helpline_section}

Simple language, culturally appropriate for Northeast India. No medical jargon.
"""
        )

        chain  = prompt | llm
        result = chain.invoke({
            "disease":          disease,
            "probability":      round(probability, 1),
            "state":            state,
            "water_source":     patient_data.get('water_source', 'Unknown'),
            "water_treatment":  patient_data.get('water_treatment', 'Unknown'),
            "handwashing":      patient_data.get('handwashing_practice', 'Unknown'),
            "flooding":         "Yes" if patient_data.get('flooding', 0) == 1 else "No",
            "symptoms":         symptoms_str,
            "helpline_section": helpline_section
        })

        logging.info("Health advisory generated successfully")

        if hasattr(result, 'content'): return result.content.strip()
        elif isinstance(result, list): return result[0].content.strip() if result else ""
        else: return str(result).strip()

    except Exception as e:
        raise CustomException(e, sys)