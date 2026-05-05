import json

def load_translations(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for row in data:
        key = f"{row['doc_id']}_{row['chunk_index']}"
        mapping[key] = {
            "si": row.get("translation_si", ""),
            "ta": row.get("translation_ta", "")
        }

    return mapping


def attach_translations(chunks, translation_map):
    enriched_chunks = []

    for chunk in chunks:
        key = f"{chunk['source_id']}_{chunk.get('chunk_index', 0)}"

        translations = translation_map.get(key)

        if translations:
            combined_text = f"""
[EN]
{chunk['text']}

[SI]
{translations.get('si', '')}

[TA]
{translations.get('ta', '')}
"""
        else:
            # fallback → only English
            combined_text = chunk["text"]

        chunk["combined_text"] = combined_text.strip()
        enriched_chunks.append(chunk)

    return enriched_chunks