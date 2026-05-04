# def chunk_docs(docs, chunk_size=800, overlap=100):
#     chunks = []

#     for doc in docs:
#         text = doc["text"]
#         start = 0
#         chunk_id = 0

#         while start < len(text):
#             end = start + chunk_size
#             piece = text[start:end]

#             chunks.append({
#                 "source_id": doc["id"],
#                 "chunk_id": f"{doc['id']}_chunk_{chunk_id}",
#                 "text": piece
#             })

#             start = end - overlap
#             chunk_id += 1

#     return chunks

# def chunk_docs(docs, chunk_size=800, overlap=100):
#     chunks = []

#     for doc in docs:
#         text = doc["text"]
#         start = 0
#         chunk_id = 0

#         while start < len(text):
#             end = start + chunk_size
#             piece = text[start:end]

#             chunks.append({
#                 "source_id": doc["id"],
#                 "chunk_id": f"{doc['id']}_chunk_{chunk_id}",
#                 "chunk_index": chunk_id,
#                 "text": piece
#             })

#             start = end - overlap
#             chunk_id += 1

#     return chunks


def chunk_docs(docs, chunk_size=800, overlap=100):
    chunks = []

    for doc in docs:
        text = doc.get("markdown", "")   # ✅ correct field
        doc_id = doc.get("doc_id")       # ✅ correct ID

        if not text or not doc_id:
            continue  # skip bad records

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            piece = text[start:end]

            chunks.append({
                "source_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                "chunk_index": chunk_index,   # ✅ VERY IMPORTANT for translations
                "text": piece
            })

            start = end - overlap

            # prevent infinite loop edge case
            if start < 0:
                break

            chunk_index += 1

    return chunks