import os
import sys
import json
import time
import pdfplumber
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from src.logger import logging
from src.exception import CustomException

load_dotenv()

# ─── LLM Setup ───────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# ─── State Helplines ─────────────────────────────────────────────
STATE_HELPLINES = {
    'Assam':             {'disease_control': '0361-2237240', 'govt_health': 'DHS Assam — 0361-2237006'},
    'Manipur':           {'disease_control': '0385-2450137', 'govt_health': 'DHS Manipur — 0385-2411447'},
    'Meghalaya':         {'disease_control': '0364-2224318', 'govt_health': 'DHS Meghalaya — 0364-2220458'},
    'Mizoram':           {'disease_control': '0389-2322694', 'govt_health': 'DHS Mizoram — 0389-2325584'},
    'Nagaland':          {'disease_control': '0370-2271697', 'govt_health': 'DHS Nagaland — 0370-2291782'},
    'Arunachal Pradesh': {'disease_control': '0360-2212624', 'govt_health': 'DHS Arunachal — 0360-2212056'},
    'Sikkim':            {'disease_control': '03592-202323', 'govt_health': 'DHS Sikkim — 03592-202439'},
    'Tripura':           {'disease_control': '0381-2415583', 'govt_health': 'DHS Tripura — 0381-2226862'},
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


# ─── Helper: Parse LLM Result ────────────────────────────────────
def _parse_result(result) -> str:
    if hasattr(result, 'content'):
        return result.content.strip()
    elif isinstance(result, list):
        item = result[0] if result else ""
        return item.content.strip() if hasattr(item, 'content') else str(item).strip()
    else:
        return str(result).strip()


# ─── Function 1: PDF Text Extract ────────────────────────────────
def extract_text_from_pdf(pdf_file) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        logging.info("PDF text extracted successfully")
        return text
    except Exception as e:
        raise CustomException(e, sys)


# ─── Function 2: Gemini se Data Parse karo ───────────────────────
def extract_data_from_pdf(pdf_file) -> dict:
    try:
        pdf_text = extract_text_from_pdf(pdf_file)

        prompt = PromptTemplate(
            input_variables=["pdf_text"],
            template="""
You are a medical data extraction assistant.
Extract the following fields from the health report and return ONLY valid JSON.
No explanation, no markdown, no extra text — only raw JSON.

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

        chain = prompt | llm
        time.sleep(5)
        result = chain.invoke({"pdf_text": pdf_text})
        result = _parse_result(result)

        if result.startswith("```"):
            parts = result.split("```")
            result = parts[1] if len(parts) > 1 else result
            if result.startswith("json"):
                result = result[4:]
        result = result.strip()

        data = json.loads(result)
        logging.info("Data extracted from PDF successfully")
        return data

    except Exception as e:
        raise CustomException(e, sys)


# ─── Function 3: Health Advisory Generate karo ───────────────────
def generate_health_advisory(disease: str, probability: float, patient_data: dict) -> str:
    try:
        symptom_cols  = ['symptom_diarrhea', 'symptom_vomiting', 'symptom_fever',
                         'symptom_abdominal_pain', 'symptom_dehydration', 'symptom_jaundice',
                         'symptom_bloody_stool', 'symptom_skin_rash']
        symptom_names = ['Diarrhea', 'Vomiting', 'Fever', 'Abdominal Pain',
                         'Dehydration', 'Jaundice', 'Bloody Stool', 'Skin Rash']
        present_symptoms = [
            symptom_names[i] for i, col in enumerate(symptom_cols)
            if patient_data.get(col, 0) == 1
        ]
        symptoms_str = ", ".join(present_symptoms) if present_symptoms else "None"

        state      = patient_data.get('state', 'Assam')
        helplines  = STATE_HELPLINES.get(state, STATE_HELPLINES['Assam'])
        specialist = DISEASE_SPECIALIST.get(disease, 'General Physician')

        helpline_section = ""
        if probability >= 70.0 and disease != 'No_Disease':
            helpline_section = f"""

7. HELPLINE AND DOCTOR RECOMMENDATION (Confidence >= 70%)
   ** High confidence prediction — immediate medical consultation recommended **

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
You are a public health advisor following WHO (World Health Organization) guidelines
for waterborne disease management in Northeast India.

Patient Assessment:
- Predicted Disease    : {disease}
- Confidence           : {probability}%
- Location             : {state}, Northeast India
- Water Source         : {water_source}
- Water Treatment      : {water_treatment}
- Handwashing Practice : {handwashing}
- Flooding Event       : {flooding}
- Symptoms Present     : {symptoms}

Based on WHO guidelines, provide a structured health advisory:

1. DISEASE OVERVIEW
   (WHO definition, transmission route, burden in South/Southeast Asia — 2-3 lines)

2. IMMEDIATE ACTIONS (WHO Emergency Protocol)
   - 3-4 immediate steps to take NOW
   - Include WHO-recommended ORS if applicable

3. WATER SAFETY TIPS (WHO WASH Guidelines)
   - 3 specific tips based on water source: {water_source}

4. PREVENTION (WHO Protocol)
   - 3-4 WHO recommended preventive measures
   - Include WHO 5 moments of hand hygiene

5. WARNING SIGNS (WHO Red Flag Symptoms)
   - Symptoms requiring immediate hospitalization
   - High risk groups: children under 5, elderly, pregnant women

6. COMMUNITY ACTION (WHO One Health Approach)
   - 2 community level actions to prevent outbreak spread

{helpline_section}

Keep language simple and culturally appropriate for Northeast India.
No medical jargon. Write in English.
"""
        )

        chain = prompt | llm
        time.sleep(15)
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

        final = _parse_result(result)
        logging.info("Health advisory generated successfully")
        return final

    except Exception as e:
        raise CustomException(e, sys)