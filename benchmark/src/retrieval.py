# import os
# from pinecone import Pinecone
# from utils.embeddings import embed


# class PineconeRetriever:

#     def __init__(self, index_name):
#         api_key = os.getenv("PINECONE_API_KEY")
#         index_name = index_name or os.getenv("PINECONE_INDEX")

#         if not api_key:
#             raise ValueError("PINECONE_API_KEY not found in environment variables.")

#         if not index_name:
#             raise ValueError("PINECONE_INDEX not found in environment variables; provide a valid index name.")

#         self.pc = Pinecone(api_key=api_key)
#         self.index = self.pc.Index(index_name)

#     def retrieve(self, query, top_k=5):

#         # Embed query
#         query_vector = embed(query)

#         # Query Pinecone
#         results = self.index.query(
#             vector=query_vector.tolist(),
#             top_k=top_k,
#             include_metadata=True
#         )

#         matches = results.get("matches", [])

#         retrieved_texts = []
#         scores = []
#         metadata_list = []

#         print("\n===== RAW PINECONE MATCHES =====\n")

#         for i, match in enumerate(matches):

#             score = match.get("score", 0)
#             meta = match.get("metadata", {})

#             print(f"\n--- Match {i} ---")
#             print("Score:", score)
#             print("Metadata keys:", list(meta.keys()))

#             # Try multiple possible text fields
#             text = (
#                 meta.get("text") or
#                 meta.get("chunk") or
#                 meta.get("content") or
#                 ""
#             )

#             if not text:
#                 print("⚠ WARNING: No text field found in metadata.")

#             retrieved_texts.append(text)
#             scores.append(score)
#             metadata_list.append(meta)

#         return retrieved_texts, scores, metadata_list

import os
from pinecone import Pinecone
from utils.embeddings import embed


class PineconeRetriever:

    def __init__(self, index_name):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = index_name or os.getenv("PINECONE_INDEX")

        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables.")

        if not index_name:
            raise ValueError("PINECONE_INDEX not found in environment variables.")

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    # -----------------------------------------------------------
    # Helper: extract best text from metadata
    # -----------------------------------------------------------
    def _extract_text(self, meta):
        return (
            meta.get("text") or
            meta.get("text_en") or
            meta.get("chunk") or
            meta.get("content") or
            ""
        )

    # -----------------------------------------------------------
    # Helper: keyword overlap score (uses metadata keywords if available)
    # -----------------------------------------------------------
    def _keyword_score(self, query, meta):
        """
        Calculate keyword overlap score using:
        1. metadata['keywords'] (preferred - from Supabase)
        2. metadata['text'] (fallback)
        """
        query_tokens = set(query.lower().split())
        query_tokens = {t for t in query_tokens if len(t) >= 3}
        
        if not query_tokens:
            return 0.0
        
        # Try metadata keywords first (from Supabase)
        keywords = meta.get("keywords", [])
        if keywords and isinstance(keywords, list):
            keyword_tokens = set([k.lower() for k in keywords])
            overlap = query_tokens.intersection(keyword_tokens)
            if overlap:
                return len(overlap) / (len(keyword_tokens) + 1e-6)
        
        # Fallback: extract from text
        text = meta.get("text", "")
        if not text:
            return 0.0
        
        text_tokens = set(text.lower().split())
        overlap = query_tokens.intersection(text_tokens)
        
        return len(overlap) / (len(query_tokens) + 1e-6)

    # -----------------------------------------------------------
    # Helper: language match bonus
    # -----------------------------------------------------------
    def _language_bonus(self, query, meta):
        """Boost score if query language matches document language"""
        # Simple heuristic: detect language from query
        # Sinhala: කියවීමේ, සිංහල
        # Tamil: வாசிப்பு, தமிழ்
        # English: default
        
        query_lower = query.lower()
        meta_lang = meta.get("lang", "en")
        
        # Check for Sinhala characters
        if any('\u0D80' <= c <= '\u0DFF' for c in query):
            return 0.1 if meta_lang == "si" else 0.0
        
        # Check for Tamil characters
        if any('\u0B80' <= c <= '\u0BFF' for c in query):
            return 0.1 if meta_lang == "ta" else 0.0
        
        # English query - slight boost for English docs
        return 0.05 if meta_lang == "en" else 0.0

    # -----------------------------------------------------------
    # Helper: category match bonus
    # -----------------------------------------------------------
    def _category_bonus(self, query, meta):
        """Boost if query matches document category"""
        # Could be extended to detect query category
        return 0.0  # Placeholder for now

    # -----------------------------------------------------------
    # Main Retrieval
    # -----------------------------------------------------------
    def retrieve(self, query, top_k=5):

        # 1️⃣ Embed query
        query_vector = embed(query)

        # 2️⃣ Pinecone search
        # results = self.index.query(
        #     vector=query_vector.tolist(),
        #     top_k=top_k * 2,   # get more → rerank later
        #     include_metadata=True
        # )

        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        results = self.index.query(
            vector=query_vector,
            top_k=top_k * 3,
            include_metadata=True
        )

        matches = results.get("matches", [])

        print("\n===== RAW PINECONE MATCHES =====\n")

        reranked = []

        for i, match in enumerate(matches):

            score = match.get("score", 0)
            meta = match.get("metadata", {})

            print(f"\n--- Match {i} ---")
            print("Score:", score)
            print("Metadata keys:", list(meta.keys()))

            text = self._extract_text(meta)

            if not text:
                print("⚠ WARNING: No text field found in metadata.")

            # 3️⃣ Keyword boost (uses metadata keywords if available)
            keyword_boost = self._keyword_score(query, meta)

            # 4️⃣ Language match bonus
            lang_bonus = self._language_bonus(query, meta)

            # 5️⃣ Calculate final score with all boosts
            # Semantic score ( Pinecone) + keyword match + language match
            final_score = score + (0.4 * keyword_boost) + lang_bonus

            print(f"Keyword boost: {keyword_boost:.3f}, Lang bonus: {lang_bonus:.3f}")
            print(f"Final score: {final_score:.3f}")

            reranked.append({
                "text": text,
                "score": final_score,
                "meta": meta
            })

        # 6️⃣ Sort by improved score
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

        # 7️⃣ Select top_k
        reranked = reranked[:top_k]

        # 8️⃣ Format output
        retrieved_texts = [r["text"] for r in reranked]
        scores = [r["score"] for r in reranked]
        metadata_list = [r["meta"] for r in reranked]

        return retrieved_texts, scores, metadata_list