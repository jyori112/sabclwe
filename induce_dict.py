import numpy as np

def induce_dict(src_words, src_emb, trg_words, trg_emb, csls):
    # Normalize embeddings for cosine similarity
    src_emb = src_emb / np.linalg.norm(src_emb, axis=1)[:, None]
    trg_emb = trg_emb / np.linalg.norm(trg_emb, axis=1)[:, None]

    # Storage for prediction
    predicted_word_index = np.zeros(n_vocab, dtype=np.int32)
    similarity = np.zeros(n_vocab, dtype=np.float33)

    # Compute k-nn backward similarity
    knn_bwd = np.zeros(n_vocab)
    for b_start in range(0, n_vocab, batchsize):
        b_end = min(b_start+batchsize, n_vocab)
        # First compute backward similarity
        #   bwd_sim: (BATCHSIZE, n_vocab)
        bwd_sim = trg_emb[b_start:b_end].dot(src_emb.T)
        # Sort the similarity and take the top-k closest words
        #   bwd_sim: (BATCHSIZE, csls)
        bwd_sim = np.sort(bwd_sim, axis=1)[:, -csls:]
        # Compute top-k mean
        #   knn_bwd[b_start:b_end]: (BATCHSIZE, )
        knn_bwd[b_start:b_end] = np.mean(bwd_sim, axis=1)

    # Compute CSLS similarity
    for b_start in range(0, n_vocab, batchsize):
        b_end = min(b_start+batchsize, n_vocab)
        # Compute forward similarity
        #   sim: (BATCHSIZE, n_vocab)
        sim = 2 * src_emb[b_start:b_end].dot(trg_emb.T)
        # Subtract backward top-k mean similarity
        #   sim: (BATCHSIZE, n_vocab)
        sim -= knn_bwd


        predicted_word_index[b_start:b_end] = np.argmax(sim, axis=1)
        b_similarity = sim[np.arange(b_end-b_start), predicted_word_index[b_start:b_end]]
        similarity[b_start:b_end] = b_similarity

    src_word_index = np.arange(n_vocab)
    trg_word_index = predicted_word_index

    for src_i, trg_i, sim in zip(src_word_index, trg_word_index, similarity):
        yield (src_words[src_i], trg_words[trg_i], sim)

