import Corpus
import os
from scipy.cluster import hierarchy


if __name__ == '__main__':
    CORPUS_ROOT = r'.\sim_data'
    corpus = Corpus.Corpus(os.path.join(CORPUS_ROOT, 'english'), text_language='en')
    names = os.listdir(r'C:\Users\51694\PycharmProjects\textClassifierPrac\sim_data\english\en')
    names = [i[:i.find('.txt')] for i in names]
    bow = corpus.build_bow()
    file_vectors = corpus.files_data(bow, 'TF-IDF')[0]
    dist = hierarchy.distance.squareform(hierarchy.distance.pdist(file_vectors, metric='cosine'))
    for i, row in enumerate(dist):
        for j, pix in enumerate(row):
            dist[i][j] = 1 - pix
    print(dist)
    out = []
    for i, line in enumerate(dist):
        sims = [(names[j], x) for j, x in enumerate(line) if x < 1]
        best10_sims = sorted(sims, key=lambda x: x[1], reverse=True)[:10]
        output_line = names[i] + '\t' + '; '.join([x[0] for x in best10_sims])
        out.append(output_line)
    print(out)
    with open('en_output.txt', 'w') as f:
        f.write('\n'.join(out))

