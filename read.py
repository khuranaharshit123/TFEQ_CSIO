# from bson import decode_file_iter

# with open("bson_file/accelerogram.bson", "rb") as f:
#     for i, document in enumerate(decode_file_iter(f)):
#         print("Document Number:", i + 1)
#         print("Type:", type(document))
#         print("Keys:", document.keys())
#         print("\nFirst Document:\n", document)
#         break


# from bson import decode_file_iter

# with open("bson_file/accelerogram.bson", "rb") as f:
#     for document in decode_file_iter(f):

#         signal = document["accelerogram"]

#         print("ID:", document["_id"])
#         print("Type:", type(signal))
#         print("Length:", len(signal))
#         print("First 10 values:", signal[:10])

#         break


# from bson import decode_file_iter

# with open("bson_file/header.bson", "rb") as f:
#     for document in decode_file_iter(f):
#         print(document.keys())
#         print(document)
#         break


# from bson import decode_file_iter

# with open("bson_file/accelerogram.bson", "rb") as f:

#     for document in decode_file_iter(f):

#         print("ID:", document["_id"])
#         print("Number of samples:", len(document["accelerogram"]))

#         break



# from bson import decode_file_iter

# count = 0

# with open("bson_file/accelerogram.bson", "rb") as f:
#     for document in decode_file_iter(f):
#         print(document["_id"])

#         count += 1
#         if count == 10:
#             break


from bson import decode_file_iter

with open("bson_file/accelerogram.bson", "rb") as f:
    doc = next(decode_file_iter(f))

print(doc.keys())

for k, v in doc.items():
    print(k, type(v))

print("\nFirst sample type:", type(doc["accelerogram"][0]))
print("First sample:", doc["accelerogram"][0])