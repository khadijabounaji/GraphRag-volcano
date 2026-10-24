#!/usr/bin/env python3
import re
import unicodedata

def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8').lower()
    return s.replace("°e", "e").replace("°", "")

def is_offtopic(question: str) -> bool:
    """
    Détecte si une question est hors du domaine géologique.
    """
    q = question.lower().strip()
    
    # Nettoyer les caractères spéciaux pour la détection de mots
    q_clean = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', q).strip()
    
    # Questions trop courtes ou sans sens
    if len(q_clean.split()) < 2:
        print(f"REJECT: Too few words ({len(q_clean.split())})")
        return True
    
    # Détection de questions invalides : contient principalement des caractères spéciaux
    non_alpha_count = sum(1 for c in q if not c.isalpha() and c != ' ')
    total_chars = len(q.replace(" ", ""))
    if total_chars > 0 and non_alpha_count / total_chars > 0.5:  # Plus de 50% de caractères spéciaux
        print(f"REJECT: Too many special chars ({non_alpha_count}/{total_chars})")
        return True
    
    # Mots-clés hors domaine géologique - ÉTENDU
    # NOTE: "pourquoi" a été RETIRÉ - c'est une question géologique valide !
    offtopic_words = [
        "yes", "no", "yep", "nope", "maybe", "ok", "oui", "non",
        "lol", "haha", "???", "!!", "test", "hello", "hi",
        "bonjour", "salut", "merci", "khadija", "mohamed", "ahmed",
        "qui es-tu", "ton nom", "comment tu", "aiut", "help"
    ]
    for w in offtopic_words:
        if re.search(r'\b' + re.escape(w) + r'\b', q_clean):
            print(f"REJECT: Found offtopic word '{w}'")
            return True
    
    # Pas un seul mot du domaine géologique
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
    
    # Si question longue (>4 mots) mais aucun mot du domaine → probablement hors-sujet
    if len(q_clean.split()) > 4 and not has_domain:
        print(f"REJECT: Long question with no domain words ({len(q_clean.split())} words)")
        print(f"  Words: {q_clean.split()}")
        return True
    
    print(f"ACCEPT: Question is on-topic")
    return False

# Test
question = "Qu'est-ce que la méthode ZDD géochronologique ?"
result = is_offtopic(question)
print(f"\nFinal result: is_offtopic('{question}') = {result}")
