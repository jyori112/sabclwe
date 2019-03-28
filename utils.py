import numpy as np

def read_emb(fobj, n_vocab=None):
    total_vocab, dim = fobj.readline().split()
    total_vocab = int(total_vocab)
    dim = int(dim)

    if n_vocab is None:
        n_vocab = total_vocab

    matrix = np.empty((n_vocab, dim), dtype=np.float32)
    words = []

    i = 0

    while True:
        if i >= n_vocab:
            break

        line = fobj.readline()


        word, vec_str = line.rstrip().split(' ', 1)

        if not word.strip():
            continue

        words.append(word)
        #print(vec_str)
        matrix[i] = np.fromstring(vec_str, sep=' ', dtype=np.float32)

        i += 1

    return words, matrix

