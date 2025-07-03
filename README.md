# Knowledge_Graph_Embeddinggs_pyRDF2vec
# pyRDF2Vec: RDF2Vec Embedding Generator for GO and HP Ontologies

This project generates knowledge graph embeddings for the Gene Ontology (GO) and Human Phenotype Ontology (HP) using [pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec). The script parses both ontologies, trains embedding models, and outputs a single merged JSON file containing dense vector representations for each GO and HP entity.

## Features

- **Combines multiple ontologies:** Merges GO and HP into a single file for unified embedding.
- **Efficient entity extraction:** Extracts all entities (URIs) from both ontologies.
- **pyRDF2Vec-based training:** Learns embeddings using random walks and Word2Vec.
- **Single merged output:** All entity embeddings are saved in `PYRDF2vec.json` as flat lists.
- **Full reproducibility:** Clear file paths, progress logs, and debugging statements.

## Important Note
1 : You need to create files 'output.log' which can save output.
2 : You need to create folder named 'embeddings' which can save the json format outtput file.
  

---
## Requirements

- **Python 3.7+** (tested with Python 3.9)
- [rdflib](https://pypi.org/project/rdflib/)
- [pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec) (`pip install pyrdf2vec==0.2.3`)
- [gensim](https://pypi.org/project/gensim/)

Install all dependencies:

```bash
you have to install dependencies for this mannually
pip install -m (Dependency name)
