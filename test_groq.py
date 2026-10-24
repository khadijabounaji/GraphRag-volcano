import urllib.request, json

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

payload = json.dumps({
    "model": GROQ_MODEL,
    "messages": [
        {"role": "system", "content": "Tu es un géologue expert."},
        {"role": "user", "content": "En 1 phrase: Quelle relation y a-t-il entre Caldera et Rift-zone N030 si Caldera CONTROLE_STRUCTURAL Rift-zone N030 ?"}
    ],
    "temperature": 0.3,
    "max_tokens": 200
}).encode("utf-8")

req = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    },
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("SUCCESS:", data["choices"][0]["message"]["content"].strip())
except Exception as e:
    print(f"ERREUR: {type(e).__name__}: {e}")
