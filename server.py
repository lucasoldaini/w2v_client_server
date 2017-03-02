import gensim
import numpy
import pickle
from argparse import ArgumentParser
from time import time

try:
    from MinimalServer import run_server
except ImportError:
    from .MinimalServer import run_server


class W2VServer:
    def __init__(self, path, binary, gensim_class):
        start = time()
        if gensim_class:
            try:
                self.w2v = gensim.models.Word2Vec.load(path)
            except UnicodeDecodeError:
                with open(path, 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    self.w2v = u.load()
        else:
            self.w2v = gensim.models.Word2Vec.load_word2vec_format(
                path, binary=binary
            )


        delta = time() - start
        print('[info] model loaded in {:.2f} s'.format(delta))

        self.oov_word = numpy.zeros(self.w2v.vector_size)

    def set_oov(self, oov_word):
        self.oov_word = oov_word

    def get_oov(self):
        return self.oov_word

    def similar_by_vector(self, vector, topn=10):
        return self.w2v.similar_by_vector(vector=vector, topn=topn)

    def get_vector_size(self):
        return self.w2v.vector_size

    def _get_term(self, term):
        if term in self.w2v.vocab:
            return self.w2v[term]
        else:
            return self.oov_word

    def get(self, data):
        if isinstance(data, str):
            return self._get_term(data)
        else:
            return [self.get(term) for term in data]


def run_w2v_server(opts):
    w2v = W2VServer(
        path=opts.path, binary=opts.binary,
        gensim_class=opts.gensim_class)

    run_server(
        w2v, host=opts.host, port=opts.port, buffersize=4096)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('-b', '--binary', action='store_true')
    ap.add_argument('-g', '--gensim-class', action='store_true')
    ap.add_argument('-H', '--host', default='localhost')
    ap.add_argument('-P', '--port', default=7443, type=int)

    opts = ap.parse_args()
    run_w2v_server(opts)
