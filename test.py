import chromadb

db_path = 'VECTOR_STORE'
collection_name = 'product'

client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name=collection_name)

# Lấy documents và metadata
data = collection.get(include=["documents", "metadatas"])
documents_list = data.get("documents", [])
metadatas_list = data.get("metadatas", [])

print(metadatas_list)


  
