import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("SUPABASE_URL").strip()
key = os.getenv("SUPABASE_KEY").strip()

client = create_client(url, key)
# Try a simple select to see if the connection works at all
try:
    res = client.table("evaluations").select("*", count="exact").limit(1).execute()
    print("Connection Successful!")
except Exception as e:
    print(f"Connection Failed: {e}")