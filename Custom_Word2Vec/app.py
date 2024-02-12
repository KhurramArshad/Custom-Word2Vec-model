from DataPreprocess import DataPipeline
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
import gensim

def clean_data_and_get_embedding(path, text_column_name):
    story = []
    obj = DataPipeline(path, text_column_name)
    raw_sent = obj.run()
    for sent in raw_sent:
        story.append(simple_preprocess(sent))
    print(story)
    print(len(story))
    model = gensim.models.Word2Vec(window=5, min_count=3, vector_size=150)
    vocab = model.build_vocab(story)
    # print("Vocab ==> ", vocab)
    model.train(story, total_examples=model.corpus_count, epochs=5)
    model.save("word2vec.model")
    # print(model.wv["attached"])
    return "Embedding model is saved successfully"


path = "../data/sample.csv"
text_column_name = "text"

result = clean_data_and_get_embedding(path, text_column_name)
print(result)


