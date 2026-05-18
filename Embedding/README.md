# Representasi & Embedding Teks

**Mata kuliah:** Pemrosesan Bahasa Alami Berbasis Transformer  
**Program Studi Sains Data · Fakultas Sains · Institut Teknologi Sumatera**

Folder ini mempelajari evolusi **representasi kata** dari model statistik klasik hingga embedding berbasis **Transformer** (BERT, ALBERT).

| Modul | Skrip | Topik |
|-------|--------|--------|
| Word2Vec Skip-gram | `001-skipgram-word2vec.py` | [skipgram_word2vec/README.md](skipgram_word2vec/README.md) |
| BERT dari nol | `002-bert.py` | [bert/README.md](bert/README.md) |
| ALBERT dari nol | `003-albert.py` | [albert/README.md](albert/README.md) |
| NPLM (klasik) | `004-NPLM.py` | [nplm/README.md](nplm/README.md) |

**Sumber asli:** [shawroad/NLP_pytorch_project](https://github.com/shawroad/NLP_pytorch_project) — folder `Embedding`.

## Peta konsep

```
NPLM (1986)  →  Word2Vec (2013)  →  BERT/ALBERT (2018+)
   trigram        vektor padat         kontekstual + pra-pelatihan
```
