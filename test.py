import chromadb

client = chromadb.PersistentClient(path="VECTOR_STORE")

print(client.list_collections())