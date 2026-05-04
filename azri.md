@118764486033646 

endpoint and sample script ,
import openai

# Initialize the client pointing to your specific Vast.ai instance
# We use the external IP and the external port (53138)
client = openai.OpenAI(
    api_key="YOUR_CUSTOM_SECRET_KEY", # Use the exact key you set in your startup command
    base_url="http://1.208.108.242:53138/v1" 
)

# The exact system prompt with your rules and JSON requirements
system_prompt = """You are a specialized healthcare routing assistant for Sri Lankan queries.
Your task is to classify and extract routing metadata from the user's query.

Strict Rules:
1. Never provide medical answers, diagnoses, or prescriptions.
2. Never provide dosage instructions or emergency action instructions.
3. The rationale must be for classification only, not medical reasoning.
4. You must output ONLY a valid JSON object. Do not include any conversational text outside the JSON.

The JSON object must contain exactly these keys: "language", "intent", "keywords", "rationale", and "english_translation_or_summary".

intent categorries are : 

"Emergency_Triage",
"Symptom_Information",
"Disease_Information",
"Facility_Locator",
"Provider_Locator",
"Appointment_Booking",
"Medication_Information",
"Vaccine_Information",
"Test_Diagnostics",
"Treatment_Procedure",
"Cost_Insurance",
"General_Health_Education",
"Non_Medical",
"Unclear"

"""

# Example of intent classifier model output from user query
{
"language": "Tamil English Code Mixed",
"intent": "Treatment_Procedure",
"user_query": "Colombo hospital-la enoda knee replacement
surgery mudinjuthu. Ippo physiotherapy treatment eppadi
"keywords": [
"physiotherapy",
"treatment",
"procedure",
"knee replacement",
"surgery",
"Colombo"]
"rationale": "The user is asking 'how' ('eppadi') a 'physiotherapy
treatment' will proceed after their surgery. This is a query about
the process of a medical procedure, classifying it as
Treatment_Procedure. The sentence mixes Tamil ('enoda',
'mudinjuthu', 'Ippo', 'nadakkum') and English ('Colombo
hospital', 'knee replacement surgery', 'physiotherapy
treatment'), making it Tamil English Code Mixed.",

"english_translation_or_summary": "My knee replacement
surgery at the Colombo hospital is finished. Now, how will the
physiotherapy treatment proceed?"

}


# Make the API call
response = client.chat.completions.create(
    model="research-mates/Qwen3-4B-SL-Health-Router-v1",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Mata Suwa Seriya eken dunna Panadol eka BP behet ekka gaththata kamak nadda?"}
    ],
    temperature=0.0, # Keep at 0.0 to force strict, deterministic JSON output
    max_tokens=300
)

what happen is when user gives input it directly goes to new intent classifier agent. intent classifier generate requred strutured output. if "intent" =
"Emergency_Triage" directly gives final asnwer of 

"
Emergency Contact

Ambulance Service: 1990
Children and Women Welfare Hotline:1929
General Government Helpline : 1919

"
and "intent"  is "Unclear" or non medical direct 'query classifier' agent to generate follow up question 

Rest of "intent" categories are direct to to medical info


when query classifier agent generate followup question it goes to user and then again user's answer passes to "intent classifier" agent. 

as this new system, current "intent classifier" need to be changed as  "query classifier"