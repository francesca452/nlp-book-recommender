# Abbiamo terminato tutti i componenti per il book-recommender
# Il vector database permette di trovare libri simili alla query
# Il text classiificator permetted di classificare i libri in categoria "fiction" e
# nonfiction", significando che gli utenti possano filtrare i libri in base alla categoria
# Abbiamo anche associato ad ogni libro probabilità di emotion
# Quello che abbiamo è quindi un buon codice e un buon dataset
# Ma rimangono comunque tali, nulla di più
# Come renderli più user-friendly? 

# Creiamo una dashboard che permetta agli utenti di utilizzare l'applicazione in modo più comodo
# Ci muoviamo ora su un file Python e abbiamo abbandonato i Notebook Jupyter

# Import libraries and depences
from doctest import OutputChecker
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import re

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

# Gradio è un pacchetto open source di Python che permette di mostrare dashboard
# appositamente pensate per modelli di machine learning
# fonte: gradio.app/guides/quickstart
# Utilizzeremo thumbnail nel book-recommender, che permette una piccola preview della 
# copertina dei libri (è una feature nel dataset)
# Il dataset da link a google books per fare vedere l'immagine di copertina, ma quello
# che di default da sono immagini di copertine con dimensioni diverse
# Chiediamo al dataset di darci la massima dimensionalità delle immagini per permettere la 
# migliore risoluzione 
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# Abbiamo dei libri senza copertina, quindi quando facciamo girare il codice otteniamo un errore
# Per risolvere, usiamo una copertina standard per libri che non ce l'hanno
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# Codice per creare il vector database, che è il cuore del book-recommender
# che è la "semantic recommendations"
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embeddings)

# Costruiamo una dfunzione che recupererà questi consigli di lettura, che farà anche
# filtraggio sulla base delle categorie e in base all' "emotion tone"
# All'inizio si prendono 50 libri e poi ne vengono restituiti dopo il filtraggio 16
# (scelto 16 per il fatto che si dimostra bene presentato nella dashboard)
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50, 
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = []

    for rec in recs:
        match = re.search(r"\b\d{10,13}\b", rec.page_content)
        if match:
            try:
                isbn_int = int(match.group())
                books_list.append(isbn_int)
            except ValueError:
                continue

    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Facciamo ora filtraggio basato sulle categorie
    # Permettiamo di leggere tutte le categorie o solo 4, che sono le principali (fiction, nonfiction,
    # children's fiction e children's nonfiction)
    # Deciso di non mettere come ricerca del sentiment il "disgust" e il "neutral" perchè
    # il primo non immagino nessuno che lo voglia cercare, e il secondo lo si usa se non viene
    # fatto un filtraggio per il sentiment sul libro da leggere

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)
    
    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Sospenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sad", ascending=False, inplace=True) 

    return books_recs


# Facciamo ora una funzione che mostri cosa fare vedere nel dashboard
def recommend_books(
    query: str,
    category: str,
    tone: str,
): 
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    # Viene fatto un loop sui consigli ritornati
    for _, row in recommendations.iterrows():
        description = row["description"]
        # Non vogliamo vedere tutta la descrizione, ma la dividiamo in parole:
        # se supera le 30 parole, permettiamo di mostrarla solo con una azione successiva
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Facciamo un ragionamento simile anche per la lista di autori:
        # se un libro ha più di un autore, sono combinati in una semicolon
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        
        # Il modo in cui mostreremo tutte queste informazioni sul libro sarà tramite
        # una caption che si aggiungerà all'immagine di copertina del libro
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    
    return results


# Ora abbiamo le funzioni necessarie a creare la dashboard
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Abbiamo tanti tempi di gradio: visita a gradio.app/guides/theming-guide
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender") # Titolo della dashboard

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")
    
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns=8, rows=2)

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs= output
    )
    

if __name__ == "__main__":
    print("Launching Gradio dashboard")
    dashboard.launch(share=True)
