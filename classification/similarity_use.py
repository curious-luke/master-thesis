from scipy import spatial
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tf_hub


def get_emb_tensors(np_vec):
    tf_vec = []
    for i in np_vec:
        tensor_t = tf.convert_to_tensor(i)
        tf_vec.append(tensor_t)
    return tf_vec


def get_cos_sim(input_emb, tf_vec, train_set):
    rf = train_set
    cos_sim_vec = []
    for e in tf_vec:
        cos_sim = 1 - spatial.distance.cosine(e, input_emb)
        cos_sim_vec.append(cos_sim)
    rf['cos_sim']= cos_sim_vec
    return rf


def most_similar(sim_results, num_results):
    sorting = sim_results.sort_values(by='cos_sim', ascending=False)
    top_hits = sorting.head(num_results)
    top_hits_df = pd.DataFrame.reset_index(top_hits)
    top_hits_df.drop(columns=['index'])
    return top_hits_df


def similarity_calc(input_text, language, USEmodel):

    if language == True: #english model
        data_en = pd.read_csv('model/hasoc_en_train.tsv', error_bad_lines=False, encoding='utf-8', sep='\t', header=0)
        train_set = pd.DataFrame(data_en)
        np_vec = np.load("model/use_embeddings_en.npy")
    else: #german model
        data_de = pd.read_csv('model/hasoc_de_train.tsv', error_bad_lines=False, encoding='utf-8', sep='\t', header=0)
        train_set = pd.DataFrame(data_de)
        np_vec = np.load("model/use_embeddings_de.npy")

    input_emb = USEmodel(input_text)

    num_results = 9 #number of top hits to display
    tf_vec = get_emb_tensors(np_vec)
    sim_results = get_cos_sim(input_emb, tf_vec, train_set)
    ranking = most_similar(sim_results, num_results)

    return ranking
