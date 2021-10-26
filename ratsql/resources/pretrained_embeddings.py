import abc
import functools
import os
import time

import bpemb
import torch
import torchtext
import requests, zipfile, io

from ratsql.resources import corenlp
from ratsql.resources import vncorenlp
from ratsql.utils import registry
from gensim.models.keyedvectors import KeyedVectors


class Embedder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, sentence):
        '''Given a string, return a list of tokens suitable for lookup.'''
        pass

    @abc.abstractmethod
    def untokenize(self, tokens):
        '''Undo tokenize.'''
        pass

    @abc.abstractmethod
    def lookup(self, token):
        '''Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None.'''
        pass

    @abc.abstractmethod
    def contains(self, token):
        pass

    @abc.abstractmethod
    def to(self, device):
        '''Transfer the pretrained embeddings to the given device.'''
        pass


@registry.register('word_emb', 'glove')
class GloVe(Embedder):

    def __init__(self, kind, lemmatize=False):
        cache = os.path.join(os.environ.get('CACHE_DIR', os.getcwd()), '.vector_cache')
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors
        self.lemmatize = lemmatize
        self.corenlp_annotators = ['tokenize', 'ssplit']
        if lemmatize:
            self.corenlp_annotators.append('lemma')

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        ann = corenlp.annotate(text, self.corenlp_annotators)
        if self.lemmatize:
            return [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
        else:
            return [tok.word.lower() for sent in ann.sentence for tok in sent.token]
    
    @functools.lru_cache(maxsize=1024)
    def tokenize_for_copying(self, text):
        ann = corenlp.annotate(text, self.corenlp_annotators)
        text_for_copying = [tok.originalText.lower() for sent in ann.sentence for tok in sent.token]
        if self.lemmatize:
            text = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
        else:
            text = [tok.word.lower() for sent in ann.sentence for tok in sent.token]
        return text, text_for_copying

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        i = self.glove.stoi.get(token)
        if i is None:
            return None
        return self.vectors[i]

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)

@registry.register('word_emb', 'phow2v')
class PhoW2V(Embedder):
    def __init__(self, emb_path):
        load_w2v = KeyedVectors.load_word2vec_format(emb_path, binary=False)
        load_w2v.init_sims(replace=True)
        w2v_path = os.path.join(emb_path[:emb_path.rindex("/")], 'w2v')
        load_w2v.save(w2v_path)
        self.phoemb = KeyedVectors.load(w2v_path, mmap='r')
        self.dim = self.phoemb.vector_size
        self.vocab = list(self.phoemb.key_to_index.keys())

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        ann = vncorenlp.tokenize(text)
        return [tok.lower() for sent in ann for tok in sent]

    @functools.lru_cache(maxsize=1024)
    def tokenize_for_copying(self, text):
        ann = vncorenlp.tokenize(text)
        text = [tok.lower() for sent in ann for tok in sent]
        text_for_copying = [tok.lower() for sent in ann for tok in sent]
        return text, text_for_copying

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        try:
            return torch.tensor(self.phoemb.get_vector(token))
        except:
            return None

    def contains(self, token):
        return token in self.vocab

    def to(self, device):
        self.vectors = self.vectors.to(device)

@registry.register('word_emb', 'bpemb')
class BPEmb(Embedder):
    def __init__(self, dim, vocab_size, lang='en'):
        self.bpemb = bpemb.BPEmb(lang=lang, dim=dim, vs=vocab_size)
        self.dim = dim
        self.vectors = torch.from_numpy(self.bpemb.vectors)

    def tokenize(self, text):
        return self.bpemb.encode(text)

    def untokenize(self, tokens):
        return self.bpemb.decode(tokens)

    def lookup(self, token):
        i = self.bpemb.spm.PieceToId(token)
        if i == self.bpemb.spm.unk_id():
            return None
        return self.vectors[i]

    def contains(self, token):
        return self.lookup(token) is not None

    def to(self, device):
        self.vectors = self.vectors.to(device)

