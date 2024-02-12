import gensim

# Load pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load("UTECEmbbeding.model")

print(model.wv["voltage"])
# print(model.wv["error"])
print(model.wv.most_similar("voltage"))
