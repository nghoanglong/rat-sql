import os
import sys
import requests, zipfile, io

from vncorenlp import VnCoreNLP
import requests


class VNCoreNLP:
    def __init__(self):
        if not os.environ.get('VNCORENLP_HOME'): 
            extract_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../third_party'))
            r = requests.get('https://github.com/vncorenlp/VnCoreNLP/archive/master.zip')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(extract_dir)
            
            vncorenlp_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '../../third_party/VnCoreNLP-master'))
            os.environ["VNCORENLP_HOME"] = vncorenlp_dir

        if not os.path.exists(os.environ['VNCORENLP_HOME']):
            raise Exception(
                f'''Please install Stanford CoreNLP and put it at {os.environ['CORENLP_HOME']}.

                Direct URL: http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
                Landing page: https://stanfordnlp.github.io/CoreNLP/''')
        self.client = VnCoreNLP(vncorenlp_dir + '/VnCoreNLP-1.1.1.jar')

    def __del__(self):
        pass

    def tokenize(self, text):
        try:
            result = self.client.tokenize(text)
        except (self.client.PermanentlyFailedException,
                requests.exceptions.ConnectionError) as e:
            print('\nWARNING: VnCoreNLP connection timeout. Recreating the server...', file=sys.stderr)
            self.client.stop()
            self.client.start()
            result = self.client.tokenize(text)

        return result


_singleton = None


def tokenize(text):
    global _singleton
    if not _singleton:
        _singleton = VNCoreNLP()
    return _singleton.tokenize(text)
