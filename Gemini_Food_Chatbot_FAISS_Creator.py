import pandas as pd
import ast
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

CSV_PATH = r"zomato.csv"
FAISS_INDEX_DIR = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Read CSV
df = pd.read_csv(CSV_PATH)

df = df.iloc[:10]

# Parse reviews_list safely
def parse_reviews(reviews):
    try:
        parsed = ast.literal_eval(reviews)
        return [str(r[1]) for r in parsed if len(r) > 1]
    except Exception:
        return []

# Create documents with metadata
documents = []
for _, row in df.iterrows():
    reviews = parse_reviews(row["reviews_list"])
    for review in reviews:
        doc = Document(
            page_content=review,
            metadata={
                "url": row["url"],
                "address": row["address"],
                "name": row["name"],
                "online_order": row["online_order"],
                "book_table": row["book_table"],
                "rate": row["rate"],
                "votes": row["votes"],
                "phone": row["phone"],
                "location": row["location"],
                "rest_type": row["rest_type"],
                "dish_liked": row["dish_liked"],
                "cuisines": row["cuisines"],
                "approx_cost_for_two_people": row["approx_cost(for two people)"],
                "menu_item": row["menu_item"],
                "listed_in_type": row["listed_in(type)"],
                "listed_in_city": row["listed_in(city)"]
            }
        )
        documents.append(doc)

# Create FAISS index
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = FAISS.from_documents(documents, embedding)

# Save FAISS index
vectordb.save_local(FAISS_INDEX_DIR)

print(f"✅ FAISS index created with {len(documents)} documents including metadata.")
