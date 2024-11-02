
import os
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, WeightedRanker
)
from milvus_model.hybrid import BGEM3EmbeddingFunction  # Ensure this module is available
from tqdm import tqdm
from pymilvus import MilvusClient
import pickle
import json

def check_collection_info(client, collection_name,collection):
    """
    Check the count of vectors in the specified collection.

    Args:
        client (MilvusClient): The Milvus client to use for querying.
        collection_name (str): The name of the collection to check.

    Returns:
        int: The count of vectors in the collection.
    """

       # Check if the collection exists
     # Enhanced print statements for collection details

    
    print(f"Collection Name: {collection.name}")
    print(f"Collection Schema: {collection.schema}")

    collection_info = collection.describe()
    print(f"Collection Info: {collection_info}")

    result = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=["count(*)"],
    )

    count = result[0]['count(*)']
    print(f"Number of sparse vectors: {count}")
    print(f"Number of dense vectors: {count}")
    


client = MilvusClient(uri="http://192.168.1.96:19530")

to_remove = [
'context_path_chunks_all_XXX_only_noblogs',
'context_path_chunks_chunk_XX_only_noblogs',
'context_path_chunks_chunk_X_only_noblogs',
'context_path_chunks_chunk_only_noblogs',
'context_path_chunks_new_XXXX_chunk_only',
'context_path_chunks_new_XXX_chunk_only',
'context_path_chunks_new_XX_chunk_only',
'context_path_chunks_new_X_chunk_only',
'context_path_chunksXXXXXXXX'
]
for i in to_remove:
    client.drop_collection(i)
for i in sorted(client.list_collections()):
    print(i)
    # print()
    # res = client.query(
    #     collection_name=i,
    #     filter="",
    #     output_fields=["text", "metadata"],
    #     limit=1,
    # )

    # print(res)
    # print('------------')
'''
context_path_chunks_all_XXXXX_only_noblogs
context_path_chunks_chunk_XXX_only_noblogs
context_path_chunks_new_XXXXX_chunk_only
----

'''