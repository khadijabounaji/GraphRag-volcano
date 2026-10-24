import re
import unicodedata

def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8').lower()
    return s.replace("°e", "e").replace("°", "")

question = "Qu'est-ce que la méthode ZDD géochronologique ?"
q = question.lower().strip()
q_clean = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', q).strip()

print(f"Original: {question}")
print(f"Lower: {q}")
print(f"Clean: {q_clean}")
print(f"Words: {q_clean.split()}")

non_alpha_count = sum(1 for c in q if not c.isalpha() and c != ' ')
total_chars = len(q.replace(" ", ""))
print(f"Non-alpha: {non_alpha_count}/{total_chars} = {non_alpha_count/total_chars*100:.1f}%")

# Domain words check
domain_words = [
    "piton", "neiges", "caldera", "rift", "basalte",
    "phonolite", "géologie", "volcanique", "hydrothermal",
    "magma", "éruption", "formation", "roche", "mineral",
    "cilaos", "mafate", "salazie", "zdd", "u-pb", "ar-ar",
    "thermochronologie", "géochronologie", "syénite", "gabbro",
    "chunk", "document", "thèse", "méthode", "analyse",
    "relation", "structure", "faille", "brèche", "cirque"
]
has_domain = any(w in q_clean for w in domain_words)
print(f"Has domain words: {has_domain}")
for w in domain_words:
    if w in q_clean:
        print(f"  - Found: {w}")
