import unicodedata

def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8').lower()
    return s.replace('°e', 'e').replace('°', '')

def extract_entities_from_query(question: str) -> list:
    known_entities = [
        'Piton des Neiges', 'Caldera', 'Syenite', 'Cilaos', 'Mafate', 'Salazie',
        'Basalte', 'Rift N030', 'Breche Orangers', 'Phonolite', 'Source Thermale', 'Gabbro'
    ]
    synonyms = {
        'rift zone n030': 'Rift N030',
        'rift n030': 'Rift N030',
        'zone n030': 'Rift N030',
        'n030': 'Rift N030',
        'n030e': 'Rift N030',
        'caldeira': 'Caldera',
        'caldera': 'Caldera'
    }
    
    q_norm = normalize_string(question)
    found = set()
    
    for syn, canonical in synonyms.items():
        if syn in q_norm:
            found.add(canonical)
            
    for entity in known_entities:
        e_norm = normalize_string(entity)
        if e_norm in q_norm or all(word in q_norm for word in e_norm.split()):
            found.add(entity)
            
    return list(found)

print("Norm:", normalize_string('Quelle est la relation entre caldera et rift zone N030°E ?'))
print(extract_entities_from_query('Quelle est la relation entre caldera et rift zone N030°E ?'))
