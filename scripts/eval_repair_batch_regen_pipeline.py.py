#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

DOC_ROOT = Path(r"D:\SL_Medical_Corpus")
OUT_DIR = DOC_ROOT / "data" / "7_gold_standard" / "repair_work"
OUT_BATCH = OUT_DIR / "gold_eval_repair_batch_input_v2.jsonl"

SYSTEM = (
    "You are a helpful assistant. Use only the provided document content. "
    "Do not invent facts. If the answer is not in the text, say you don't know."
)

TARGET_LANGS = (
    ["en"] * 22 +
    ["si"] * 22 +
    ["ta"] * 22
)

DOCS = [
    {
        "doc_id": "d3400101311b00505a8eb7fe38d9e5bad918af8440d0468467f184ebab009ed3",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "EPID_Immunization_p0.md",
        "site": "EPID_Immunization",
    },
    {
        "doc_id": "d7962dd6a65ea7209cbd8b55deb3118d4c42b9b9e0b1a575aa2fb55f9b7f1d80",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "EPID_Immunization_p4.md",
        "site": "EPID_Immunization",
    },
    {
        "doc_id": "a31b329cae73d9cb4a96d816b2ec4eff274060e917650a02f9363f09d1186526",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "FHB_FamilyHealthBureau_p10.md",
        "site": "FHB_FamilyHealthBureau",
    },
    {
        "doc_id": "3fdc310db2f6233b7322cb92b0e62ef2310f87d598a560d28c71d432daa80ec4",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "FHB_FamilyHealthBureau_p11.md",
        "site": "FHB_FamilyHealthBureau",
    },
    {
        "doc_id": "dd6847ac3301c659721390774efb7a98813eba99b24c0acca8fcd896c96111d1",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "FHB_FamilyHealthBureau_p14.md",
        "site": "FHB_FamilyHealthBureau",
    },
    {
        "doc_id": "85e6218b636032530b5559fc56ca279c41249011c86f1bd71ed783c814e9447c",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "FHB_FamilyHealthBureau_p2.md",
        "site": "FHB_FamilyHealthBureau",
    },
    {
        "doc_id": "565485a3061a3f33b1b8c746b919136531e3197d6b5bcb2ec69f5489fe6d9f9c",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "FHB_FamilyHealthBureau_p3.md",
        "site": "FHB_FamilyHealthBureau",
    },
    {
        "doc_id": "25fbd39a3a8426f565bb406653cab147080668b899ca573169c8efc246f4d37e",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "FHB_FamilyHealthBureau_p5.md",
        "site": "FHB_FamilyHealthBureau",
    },
    {
        "doc_id": "4ba69873c056d32b2bd4ed480deb86f04f61cb5ccfbf9ce183edfb6079ad4b18",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "HPB_HealthTopics_p6.md",
        "site": "HPB_HealthTopics",
    },
    {
        "doc_id": "8709ea8e088c85468be4e1ecbf4ea8cb41899d09a777f7e61986b798cb620a35",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "HPB_HealthTopics_p7.md",
        "site": "HPB_HealthTopics",
    },
    {
        "doc_id": "e0a9f022bb40a8afa6935d9cb336a0f486c6b6e8ea77a031a1fc823cabfa13b2",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "HPB_HealthTopics_p9.md",
        "site": "HPB_HealthTopics",
    },
    {
        "doc_id": "0b9b7a8406296298f889d88eb2cb2db3b93c92c20e86a91886c8571f1daec891",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "MOH_Main_p0.md",
        "site": "MOH_Main",
    },
    {
        "doc_id": "c5d129b8da132b29c338b1cf6b14992ea0bdeb625ed08dd7c4bdb61c062ca83f",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "MOH_Main_p11.md",
        "site": "MOH_Main",
    },
    {
        "doc_id": "fc01985ce59618a2f3c02b029ce78bce7456da5604642ed20386c3bca7894421",
        "clean_path": DOC_ROOT / "data" / "1_cleaned" / "MOH_Main_p3.md",
        "site": "MOH_Main",
    },
]

GEN_PROMPT = """You are creating GOLD benchmark records for a multilingual medical instruction dataset.

Generate exactly ONE benchmark-quality JSON record from the provided source context.

Hard requirements:
- Use ONLY the provided context.
- Do NOT invent facts.
- Create a medically meaningful short factual QA example only.
- Do NOT ask about page layout, result counts, navigation labels, buttons, menus, website UI, or generic metadata.
- Prefer questions about diseases, vaccines, treatments, symptoms, recommendations, definitions, eligibility, counts, schedules, named programmes, or public health guidance.
- The answer must be concise and fully grounded in the text.
- The second assistant message must be an exact quote copied verbatim from the context.
- The first user message MUST include both the question and the full context in this exact style:

<Question text>

Context:
<full context text>

Return exactly ONE JSON object with fields:
id, doc_id, lang, task, style, split, messages

Field requirements:
- task = "short_answer_qa"
- split = "gold_eval"
- lang = "{lang}"
- doc_id = "{doc_id}"
- style = "gold_benchmark"

Message structure must be exactly:
1. system
2. user
3. assistant
4. user
5. assistant

Role content rules:
- system: "You are a helpful assistant. Use only the provided document content. Do not invent facts. If the answer is not in the text, say you don't know."
- first user: question in target language + blank line + Context: + full context
- first assistant: short answer only, in target language
- second user: ask for exact supporting line(s) in target language
- second assistant: exact quote from context only, verbatim, no explanation

Language rules:
- If lang = "en", user and first assistant must be in English.
- If lang = "si", user and first assistant must be in Sinhala.
- If lang = "ta", user and first assistant must be in Tamil.
- The evidence quote in the final assistant message must remain exactly as written in the source context, even if the context is English.

Context:
{context}

Return only the JSON object.
"""

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    requests = []

    for i, lang in enumerate(TARGET_LANGS):
        doc = DOCS[i % len(DOCS)]
        context = read_text(doc["clean_path"])[:12000]

        custom_id = f"gold_repair_v2_{lang}_{i:03d}"

        user_prompt = GEN_PROMPT.format(
            lang=lang,
            doc_id=doc["doc_id"],
            context=context,
        )

        req = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-5",
                "input": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
            },
        }
        requests.append(req)

    with OUT_BATCH.open("w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Wrote batch input ->", OUT_BATCH)
    print("Requests:", len(requests))
    print("Language counts:",
          {"en": TARGET_LANGS.count("en"), "si": TARGET_LANGS.count("si"), "ta": TARGET_LANGS.count("ta")})

if __name__ == "__main__":
    main()