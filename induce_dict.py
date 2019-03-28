import click
import numpy as np
import cupy
import utils
import sys
from tqdm import tqdm

if cupy.cuda.is_available():
    xp = cupy
else:
    xp = np

@click.command()
@click.argument('src_emb_path', type=click.Path(exists=True))
@click.argument('trg_emb_path', type=click.Path(exists=True))
@click.option('--csls', type=int, default=10)
@click.option('--batchsize', type=int, default=1000)
@click.option('--n-vocab', type=int, default=20000)
def main(src_emb_path, trg_emb_path, csls, batchsize, n_vocab):
    with open(src_emb_path) as f:
        src_words, src_emb = utils.read_emb(f, n_vocab)

    with open(trg_emb_path) as f:
        trg_words, trg_emb = utils.read_emb(f, n_vocab)

    if cupy.cuda.is_available():
        src_emb = xp.array(src_emb)
        trg_emb = xp.array(trg_emb)

    for src_word, trg_word, sim in induce_dict(src_words, src_emb, trg_words, trg_emb, csls, batchsize):
        print("{}\t{}\t{:f}".format(src_word, trg_word, sim))

def induce_dict(src_words, src_emb, trg_words, trg_emb, csls, batchsize):
    n_vocab = len(src_words)

    # Normalize embeddings for cosine similarity
    src_emb = src_emb / xp.linalg.norm(src_emb, axis=1)[:, None]
    trg_emb = trg_emb / xp.linalg.norm(trg_emb, axis=1)[:, None]

    # Storage for prediction
    predicted_word_index = xp.zeros(n_vocab, dtype=xp.int32)
    similarity = xp.zeros(n_vocab, dtype=xp.float32)

    # Compute k-nn backward similarity
    knn_bwd = xp.zeros(n_vocab)
    for b_start in tqdm(range(0, n_vocab, batchsize), desc="Step 1: Backward Similarity"):
        b_end = min(b_start+batchsize, n_vocab)
        # First compute backward similarity
        #   bwd_sim: (BATCHSIZE, n_vocab)
        bwd_sim = trg_emb[b_start:b_end].dot(src_emb.T)
        # Sort the similarity and take the top-k closest words
        #   bwd_sim: (BATCHSIZE, csls)
        bwd_sim = xp.sort(bwd_sim, axis=1)[:, -csls:]
        # Compute top-k mean
        #   knn_bwd[b_start:b_end]: (BATCHSIZE, )
        knn_bwd[b_start:b_end] = xp.mean(bwd_sim, axis=1)

    # Compute CSLS similarity
    for b_start in tqdm(range(0, n_vocab, batchsize), desc="Step 2: Compute CSLS"):
        b_end = min(b_start+batchsize, n_vocab)
        # Compute forward similarity
        #   sim: (BATCHSIZE, n_vocab)
        sim = 2 * src_emb[b_start:b_end].dot(trg_emb.T)
        # Subtract backward top-k mean similarity
        #   sim: (BATCHSIZE, n_vocab)
        sim -= knn_bwd


        predicted_word_index[b_start:b_end] = xp.argmax(sim, axis=1)
        b_similarity = sim[xp.arange(b_end-b_start), predicted_word_index[b_start:b_end]]
        similarity[b_start:b_end] = b_similarity

    src_word_index = xp.arange(n_vocab)
    trg_word_index = predicted_word_index

    if cupy.cuda.is_available():
        src_word_index = xp.asnumpy(src_word_index)
        trg_word_index = xp.asnumpy(trg_word_index)
        similarity = xp.asnumpy(similarity)

    for src_i, trg_i, sim in zip(src_word_index, trg_word_index, similarity):
        yield (src_words[int(src_i)], trg_words[int(trg_i)], sim)


if __name__ == '__main__':
    main()
