MEDICAL_TOOLS = [
    {
        "name": "hospital_lookup",
        "description": "Searches for Sri Lankan hospital contact info, available wards, and specialist doctors.",
        "parameters": {
            "type": "object",
            "properties": {
                "hospital_name": {"type": "string", "description": "Name of the hospital (e.g., Karapitiya, NHSL)"},
                "service": {"type": "string", "description": "Specific ward or specialty (e.g., Cardiology, OPD)"}
            },
            "required": ["hospital_name"]
        }
    },
    {
        "name": "drug_index_search",
        "description": "Queries the NMRA database for registered drugs, dosage forms, and safety warnings.",
        "parameters": {
            "type": "object",
            "properties": {
                "generic_name": {"type": "string", "description": "The chemical name of the drug (e.g., Paracetamol)"}
            },
            "required": ["generic_name"]
        }
    }
]