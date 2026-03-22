from weaviate_client import WeaviateClientManager

client = WeaviateClientManager(mode="cloud").get_client()

class_name = "MRInsights"

if client.schema.exists(class_name):
    client.schema.delete_class(class_name)
    print(f"Deleted class: {class_name}")
else:
    print("Class not found")