# Sono terminati tutti i passaggi per rendere funzionante il book-recommender
# - Il vector database permette di trovare libri simili alla query
# - Il text classificator ha permesso la catalogazione di libri in categoria Fiction e NonFiction
# - Ogni libro ha una probabilità di sentiment associato

# L'utente potrà usufruire dell'applicazione tramite interfaccia
# Definisco una dashboard che sfrutta il pacchetto Gradio (open source di Python)

# Librerie e dipendenze
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

# La feature "thumbnail" permette una preview della copertina dei libri
# I valori associati alla caratteristica sono link a Google Books, che possono rappresentare immagini di copertine con dimensioni diverse
# Cerco la massima dimensionalità delle immagini per una migliore risoluzione 
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# Alcuni libri sono senza copertina
# Per risolvere, uso una copertina standard e la sostisco al valore nullo per i campioni interessati
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# Codice per creare il vector database (già nel file vector-search.ipynb)
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory=None)

# Sono selezionati 50 libri dal book-reccomender e mostrati solo i primi 40 con match migliore
# La scelta del numero 40 sta nella specifica interfaccia della dashboard, che si presta a quel numero di libri mostrati
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50, 
    final_top_k: int = 40,
) -> pd.DataFrame:
    recs_with_scores = db_books.similarity_search_with_score(query, k=initial_top_k)
    
    books_list = []

    for rec, score in recs_with_scores:
        match = re.search(r"\b\d{10,13}\b", rec.page_content)
        if match:
            try:
                isbn_int = int(match.group())
                books_list.append((isbn_int, score))
            except ValueError:
                continue
    
    df_matches = pd.DataFrame(books_list, columns=["isbn13", "similarity_score"])
    result_df = books.merge(df_matches, on="isbn13")
    result_df = result_df.drop_duplicates(subset="isbn13")
    result_df = result_df.sort_values(by="similarity_score", ascending=True)

    books_recs = result_df.head(final_top_k)

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


# Funzione che definisce cosa è mostrato nella dashboard
def recommend_books(
    query: str,
    category: str,
    tone: str,
): 
    if not query.strip(): # Se la query è vuota o fatta di spazi
        return []

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    # Viene fatto un loop sui consigli ritornati
    for _, row in recommendations.iterrows():
        description = row["description"]
        # Mostro solo 30 parole della descrizione
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Simile anche per gli autori
        # Se un libro ha più di un autore, sono combinati in una semicolon
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        
        # Tutte queste informazioni sul libro sono mostrate in una caption che si aggiungerà all'immagine di copertina del libro
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    
    return results

# Si hanno a dosposizione tutte le funzioni necessarie a creare la dashboard

# Opzioni per il campo categorie (generi) e emotion state
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Temi di gradio su: gradio.app/guides/theming-guide
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender") # Titolo della dashboard

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")
    
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns=8, rows=5)

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs= output
    )
    

if __name__ == "__main__":
    print("Launching Gradio dashboard")
    dashboard.launch(share=True)

# L'interfaccia è visibile runnando questo file .py