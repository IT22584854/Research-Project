import os

def load_markdown_corpus(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({
                    "id": filename,
                    "text": text
                })
    return docs