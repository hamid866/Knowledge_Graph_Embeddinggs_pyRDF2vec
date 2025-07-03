import os
import json
import traceback
from rdflib import Graph
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

# ==== FILE PATHS ====
GO_FILE = "/home/hahmad/my_project_new/pyRDF2Vec-main/go-basic.owl"
HP_FILE = "/home/hahmad/my_project_new/pyRDF2Vec-main/hp.owl"
OUT_DIR = "/home/hahmad/my_project_new/pyRDF2Vec-main/embeddings"
OUT_JSON = os.path.join(OUT_DIR, "PYRDF2vec.json")

VECTOR_SIZE = 200
EPOCHS = 100

def log(msg):
    print(msg)
    with open("output.log", "a") as f:
        f.write(msg + "\n")

def check_path(p):
    if os.path.exists(p):
        log(f"✅ Path exists: {p}")
    else:
        log(f"❌ Path does NOT exist: {p}")
        raise FileNotFoundError(p)

def extract_entities_rdf(file_path, prefix):
    g = Graph()
    g.parse(file_path)
    log(f"Parsed {file_path} for entity extraction, triples: {len(g)}")
    entities = set()
    for s, p, o in g:
        for node in (s, o):
            uri = str(node)
            if uri.startswith("http") and prefix in uri:
                entities.add(uri)
    entities = list(entities)
    log(f"Extracted {len(entities)} unique entities from {prefix}")
    print(f"Sample extracted {prefix} entities:", entities[:5])
    return entities

def train_and_embed(kg_file, entities, label):
    log(f"Creating KG object for pyRDF2Vec ({label})...")
    kg = KG(kg_file, is_remote=False)
    log(f"KG created for {label}: {kg}")

    log(f"Training embeddings for {len(entities)} {label} entities. VECTOR_SIZE={VECTOR_SIZE}, EPOCHS={EPOCHS}, Walks=20, Depth=4")
    transformer = RDF2VecTransformer(
        Word2Vec(vector_size=VECTOR_SIZE, epochs=EPOCHS),
        walkers=[RandomWalker(4, 100)]
    )
    embeddings, literals = transformer.fit_transform(kg, entities)
    log(f"Training completed for {label}.")
    print(f"len({label} embeddings):", len(embeddings))
    print(f"Sample {label} embedding[0]:", embeddings[0] if embeddings else "EMPTY")
    return embeddings

def main():
    try:
        log(f"==== Current working directory: {os.getcwd()} ====")
        log("==== Checking input file existence... ====")
        check_path(GO_FILE)
        check_path(HP_FILE)
        os.makedirs(OUT_DIR, exist_ok=True)

        log("==== RDF2Vec Embedding Script Started (GO + HP) ====")

        # 1. Extract entities from both GO and HP ontology
        log("Extracting all entities from GO ontology...")
        go_entities = extract_entities_rdf(GO_FILE, "GO_")
        log("Extracting all entities from HP ontology...")
        hp_entities = extract_entities_rdf(HP_FILE, "HP_")

        # === Stats and samples ===
        log(f"Total GO entities: {len(go_entities)}")
        log(f"First 5 GO entities: {go_entities[:5]}")
        log(f"Total HP entities: {len(hp_entities)}")
        log(f"First 5 HP entities: {hp_entities[:5]}")

        # 2. Train embeddings for each set
        go_embeddings = train_and_embed(GO_FILE, go_entities, "GO")
        hp_embeddings = train_and_embed(HP_FILE, hp_entities, "HP")

        # 3. Save embeddings as JSON
        output = {}
        for ent, emb in zip(go_entities, go_embeddings):
            output[str(ent)] = [float(x) for x in list(emb)]
        for ent, emb in zip(hp_entities, hp_embeddings):
            output[str(ent)] = [float(x) for x in list(emb)]
        log(f"Output dictionary prepared, keys: {list(output.keys())[:5]}..., total: {len(output)}")

        with open(OUT_JSON, "w") as f:
            json.dump(output, f, indent=2)
        log(f"✅ Embeddings saved to {OUT_JSON}")

        log("==== Script Finished SUCCESSFULLY ====")
    except Exception as e:
        log("❌ ERROR: Exception occurred!")
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
