"""
This file bundles language identification functions.

Modifications (fork): Copyright (c) 2021, Adrien Barbaresi.

Original code: Copyright (c) 2011 Marco Lui <saffsd@gmail.com>.
Based on research by Marco Lui and Tim Baldwin.

See LICENSE file for more info.
"""

import argparse
import base64
import bz2
import json
import logging
import lzma

from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs

from wsgiref.simple_server import make_server
from wsgiref.util import shift_path_info

import numpy as np

import _pickle as cpickle
try:
    from cPickle import loads
except ImportError:
    from pickle import loads


logger = logging.getLogger(__name__)

# Convenience methods defined below will initialize this when first called.
identifier = None
MODEL_FILE = 'data/model.plzma'

# Defaults for inbuilt server
HOST = None  # leave as none for auto-detect
PORT = 9008
FORCE_WSGIREF = False

NORM_PROBS = False  # Normalize output probabilities.
# NORM_PROBS defaults to False for a small speed increase. It does not
# affect the relative ordering of the predicted classes. It can be
# re-enabled at runtime - see the readme.


def set_languages(langs=None):
    """
    Set the language set used by the global identifier.

    @param langs a list of language codes
    """
    global identifier
    if identifier is None:
        load_model()

    return identifier.set_languages(langs)


def classify(instance):
    """
    Convenience method using a global identifier instance with the default
    model included in langid.py. Identifies the language that a string is
    written in.

    @param instance a text string. Unicode strings will automatically be utf8-encoded
    @returns a tuple of the most likely language and the confidence score
    """
    global identifier
    if identifier is None:
        load_model()

    return identifier.classify(instance)


def rank(instance):
    """
    Convenience method using a global identifier instance with the default
    model included in langid.py. Ranks all the languages in the model according
    to the likelihood that the string is written in each language.

    @param instance a text string. Unicode strings will automatically be utf8-encoded
    @returns a list of tuples language and the confidence score, in descending order
    """
    global identifier
    if identifier is None:
        load_model()

    return identifier.rank(instance)


def cl_path(path):
    """
    Convenience method using a global identifier instance with the default
    model included in langid.py. Identifies the language that the file at `path` is
    written in.

    @param path path to file
    @returns a tuple of the most likely language and the confidence score
    """
    global identifier
    if identifier is None:
        load_model()

    return identifier.cl_path(path)


def rank_path(path):
    """
    Convenience method using a global identifier instance with the default
    model included in langid.py. Ranks all the languages in the model according
    to the likelihood that the file at `path` is written in each language.

    @param path path to file
    @returns a list of tuples language and the confidence score, in descending order
    """
    global identifier
    if identifier is None:
        load_model()

    return identifier.rank_path(path)


def load_model(path=None):
    """
    Convenience method to set the global identifier using a model at a
    specified path.

    @param path to model
    """
    global identifier
    logger.info('initializing identifier')
    if path is None:
        identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE)
    else:
        identifier = LanguageIdentifier.from_modelpath(path)


class LanguageIdentifier:
    """
    This class implements the actual language identifier.
    """

    # new version: speed-up
    @classmethod
    def from_pickled_model(cls, pickled_file, *args, **kwargs):
        # load data
        filepath = str(Path(__file__).parent / pickled_file)
        with lzma.open(filepath) as filehandle:
            nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = cpickle.load(filehandle)
        nb_numfeats = int(len(nb_ptc) / len(nb_pc))

        # reconstruct pc and ptc
        nb_pc = np.array(nb_pc)
        nb_ptc = np.array(nb_ptc).reshape(nb_numfeats, len(nb_pc))

        return cls(nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output, *args, **kwargs)

    # legacy methods
    @classmethod
    def from_modelstring(cls, string, *args, **kwargs):
        # load data
        nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = loads(bz2.decompress(base64.b64decode(string)))
        nb_numfeats = int(len(nb_ptc) / len(nb_pc))

        # reconstruct pc and ptc
        nb_pc = np.array(nb_pc)
        nb_ptc = np.array(nb_ptc).reshape(nb_numfeats, len(nb_pc))

        return cls(nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output, *args, **kwargs)

    @classmethod
    def from_modelpath(cls, path, *args, **kwargs):
        with open(path, 'rb') as f:
            return cls.from_modelstring(f.read(), *args, **kwargs)

    def __init__(self, nb_ptc, nb_pc, nb_numfeats, nb_classes, tk_nextmove, tk_output,
                 norm_probs=NORM_PROBS):
        self.nb_ptc = nb_ptc
        self.nb_pc = nb_pc
        self.nb_numfeats = nb_numfeats
        self.nb_classes = nb_classes
        self.tk_nextmove = tk_nextmove
        self.tk_output = tk_output

        if norm_probs:
            def norm_probs(pd):
                """
                Renormalize log-probs into a proper distribution (sum 1)
                The technique for dealing with underflow is described in
                http://jblevins.org/log/log-sum-exp
                """
                # Ignore overflow when computing the exponential. Large values
                # in the exp produce a result of inf, which does not affect
                # the correctness of the calculation (as 1/x->0 as x->inf).
                # On Linux this does not actually trigger a warning, but on
                # Windows this causes a RuntimeWarning, so we explicitly
                # suppress it.
                with np.errstate(over='ignore'):
                    # old:
                    pd = (1/np.exp(pd[None,:] - pd[:,None]).sum(1))
                    # suggested but doesn't work:
                    #pd_exp = np.exp(pd)
                    #pd = pd_exp / pd_exp.sum()
                return pd
        else:
            def norm_probs(pd):
                return pd

        self.norm_probs = norm_probs

        # Maintain a reference to the full model, in case we change our language set
        # multiple times.
        self.__full_model = nb_ptc, nb_pc, nb_classes

    def set_languages(self, langs=None):
        logger.debug("restricting languages to: %s", langs)

        # Unpack the full original model. This is needed in case the language set
        # has been previously trimmed, and the new set is not a subset of the current
        # set.
        nb_ptc, nb_pc, nb_classes = self.__full_model

        if langs is None:
            self.nb_classes = nb_classes
            self.nb_ptc = nb_ptc
            self.nb_pc = nb_pc

        else:
            # We were passed a restricted set of languages. Trim the arrays accordingly
            # to speed up processing.
            for lang in langs:
                if lang not in nb_classes:
                    raise ValueError("Unknown language code %s" % lang)

            subset_mask = np.fromiter((l in langs for l in nb_classes), dtype=bool)
            self.nb_classes = [c for c in nb_classes if c in langs]
            self.nb_ptc = nb_ptc[:, subset_mask]
            self.nb_pc = nb_pc[subset_mask]

    def instance2fv(self, text):
        """
        Map an instance into the feature space of the trained model.
        """
        # convert to binary if it isn't already the case
        if isinstance(text, str):
            text = text.encode('utf8')

        arr = np.zeros((self.nb_numfeats,), dtype='uint32')

        # Convert the text to a sequence of ascii values and
        # Count the number of times we enter each state
        state = 0
        indexes = []
        for letter in list(text):
            state = self.tk_nextmove[(state << 8) + letter]
            indexes.extend(self.tk_output.get(state, []))
        # Update all the productions corresponding to the state
        for index, value in Counter(indexes).items():
            arr[index] = value

        return arr

    def nb_classprobs(self, fv):
        # compute the partial log-probability of the document given each class
        pdc = np.dot(fv, self.nb_ptc)
        # compute the partial log-probability of the document in each class
        return pdc + self.nb_pc

    def classify(self, text):
        """
        Classify an instance.
        """
        fv = self.instance2fv(text)
        probs = self.norm_probs(self.nb_classprobs(fv))
        cl = np.argmax(probs)
        return self.nb_classes[cl], probs[cl]

    def rank(self, text):
        """
        Return a list of languages in order of likelihood.
        """
        fv = self.instance2fv(text)
        probs = self.norm_probs(self.nb_classprobs(fv))
        return [(str(k), float(v)) for (v, k) in sorted(zip(probs, self.nb_classes), reverse=True)]

    def cl_path(self, path):
        """
        Classify a file at a given path
        """
        with open(path) as f:
            retval = self.classify(f.read())
        return path, retval

    def rank_path(self, path):
        """
        Class ranking for a file at a given path
        """
        with open(path) as f:
            retval = self.rank(f.read())
        return path, retval


# Based on http://www.ubacoda.com/index.php?p=8
query_form = """
<html>
  <head>
    <meta http-equiv="Content-type" content="text/html; charset=utf-8">
    <title>Language Identifier</title>
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1.7.2/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" charset="utf-8">
      $(document).ready(function() {{
        $("#typerArea").keyup(displayType);

        function displayType(){{
          var contents = $("#typerArea").val();
          if (contents.length != 0) {{
            $.post(
              "/rank",
              {{q:contents}},
              function(data){{
                for(i=0;i<5;i++) {{
                  $("#lang"+i).html(data.responseData[i][0]);
                  $("#conf"+i).html(data.responseData[i][1]);
                }}
                $("#rankTable").show();
              }},
              "json"
            );
          }}
          else {{
            $("#rankTable").hide();
          }}
        }}
        $("#manualSubmit").remove();
        $("#rankTable").hide();
      }});
    </script>
  </head>
  <body>
    <form method=post>
      <center><table>
        <tr>
          <td>
            <textarea name="q" id="typerArea" cols=40 rows=6></textarea></br>
          </td>
        </tr>
        <tr>
          <td>
            <table id="rankTable">
              <tr>
                <td id="lang0">
                  <p>Unable to load jQuery, live update disabled.</p>
                </td><td id="conf0"/>
              </tr>
              <tr><td id="lang1"/><td id="conf1"></tr>
              <tr><td id="lang2"/><td id="conf2"></tr>
              <tr><td id="lang3"/><td id="conf3"></tr>
              <tr><td id="lang4"/><td id="conf4"></tr>
            </table>
            <input type=submit id="manualSubmit" value="submit">
          </td>
        </tr>
      </table></center>
    </form>

  </body>
</html>
"""


def application(environ, start_response):
    """
    WSGI-compatible langid web service.
    """
    try:
        path = shift_path_info(environ)
    except IndexError:
        # Catch shift_path_info's failure to handle empty paths properly
        path = ''

    if path in {'detect', 'rank'}:
        data = None

        # Extract the data component from different access methods
        if environ['REQUEST_METHOD'] == 'PUT':
            data = environ['wsgi.input'].read(int(environ['CONTENT_LENGTH']))
        elif environ['REQUEST_METHOD'] == 'GET':
            try:
                data = parse_qs(environ['QUERY_STRING'])['q'][0]
            except KeyError:
                # No query, provide a null response.
                status = '200 OK'  # HTTP Status
                response = {
                  'responseData': None,
                  'responseStatus': 200,
                  'responseDetails': None,
                }
        elif environ['REQUEST_METHOD'] == 'POST':
            input_string = environ['wsgi.input'].read(int(environ['CONTENT_LENGTH']))
            try:
                data = parse_qs(input_string)['q'][0]
            except KeyError:
                # No key 'q', process the whole input instead
                data = input_string
        else:
            # Unsupported method
            status = '405 Method Not Allowed'  # HTTP Status
            response = {
              'responseData': None,
              'responseStatus': 405,
              'responseDetails': '%s not allowed' % environ['REQUEST_METHOD']
            }

        if data is not None:
            if path == 'detect':
                pred, conf = classify(data)
                responseData = {'language': pred, 'confidence': conf}
            elif path == 'rank':
                responseData = rank(data)

            status = '200 OK'  # HTTP Status
            response = {
              'responseData': responseData,
              'responseStatus': 200,
              'responseDetails': None,
            }
    elif path == 'demo':
        status = '200 OK'  # HTTP Status
        headers = [('Content-type', 'text/html; charset=utf-8')]  # HTTP Headers
        start_response(status, headers)
        return [query_form.format(**environ)]

    else:
        # Incorrect URL
        status = '404 Not Found'
        response = {'responseData': None, 'responseStatus': 404, 'responseDetails': 'Not found'}

    headers = [('Content-type', 'text/javascript; charset=utf-8')]  # HTTP Headers
    start_response(status, headers)
    return [json.dumps(response)]


def main():
    global identifier

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--serve', action='store_true', default=False, dest='serve', help='launch web service')
    parser.add_argument('--host', default=HOST, dest='host', help='host/ip to bind to')
    parser.add_argument('--port', default=PORT, dest='port', help='port to listen on')
    parser.add_argument('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
    parser.add_argument('-m', dest='model', help='load model from file')
    parser.add_argument('-l', '--langs', dest='langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
    parser.add_argument('-r', '--remote', action="store_true", default=False, help='auto-detect IP address for remote access')
    parser.add_argument('-b', '--batch', action="store_true", default=False, help='specify a list of files on the command line')
    parser.add_argument('--demo', action="store_true", default=False, help='launch an in-browser demo application')
    parser.add_argument('-d', '--dist', action='store_true', default=False, help='show full distribution over languages')
    parser.add_argument('-u', '--url', help='langid of URL')
    parser.add_argument('--line', action="store_true", default=False, help='process pipes line-by-line rather than as a document')
    parser.add_argument('-n', '--normalize', action='store_true', default=False, help='normalize confidence scores to probability values')
    options = parser.parse_args()

    if options.verbosity:
        logging.basicConfig(level=max((5-options.verbosity)*10, 0))
    else:
        logging.basicConfig()

    if options.batch and options.serve:
        parser.error("cannot specify both batch and serve at the same time")

    # unpack a model
    if options.model:
        try:
            identifier = LanguageIdentifier.from_modelpath(options.model, norm_probs=options.normalize)
            logger.info("Using external model: %s", options.model)
        except IOError as e:
            logger.warning("Failed to load %s: %s" % (options.model, e))

    if identifier is None:
        identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=options.normalize)
        logger.info("Using internal model")

    if options.langs:
        langs = options.langs.split(",")
        identifier.set_languages(langs)

    def _process(text):
        """
        Set up a local function to do output, configured according to our settings.
        """
        return identifier.rank(text) if options.dist else identifier.classify(text)

    if options.url:
        import contextlib
        from urllib.request import urlopen
        with contextlib.closing(urlopen(options.url)) as url:
            text = url.read()
            output = _process(text)
            print(options.url, len(text), output)

    elif options.serve or options.demo:
        # from http://stackoverflow.com/questions/166506/finding-local-ip-addresses-in-python
        if options.remote and options.host is None:
            # resolve the external ip address
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("google.com", 80))
            hostname = s.getsockname()[0]
        elif options.host is None:
            # resolve the local hostname
            import socket
            hostname = socket.gethostbyname(socket.gethostname())
        else:
            hostname = options.host

        if options.demo:
            import webbrowser
            webbrowser.open('http://{0}:{1}/demo'.format(hostname, options.port))
        try:
            if FORCE_WSGIREF:
                raise ImportError
            # Use fapws3 if available
            import fapws._evwsgi as evwsgi
            from fapws import base
            evwsgi.start(hostname, str(options.port))
            evwsgi.set_base_module(base)
            evwsgi.wsgi_cb(('', application))
            evwsgi.set_debug(0)
            evwsgi.run()
        except ImportError:
            print("Listening on %s:%d" % (hostname, int(options.port)))
            print("Press Ctrl+C to exit")
            httpd = make_server(hostname, int(options.port), application)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
    elif options.batch:
        # Start in batch mode - interpret input as paths rather than content
        # to classify.
        import csv
        import os
        import sys
        import multiprocessing as mp

        def generate_paths():
            for line in sys.stdin:
                path = line.strip()
                if path and os.path.isfile(path):
                    yield path

        writer = csv.writer(sys.stdout)
        pool = mp.Pool()
        if options.dist:
            writer.writerow(['path'] + identifier.nb_classes)
            for path, ranking in pool.imap_unordered(rank_path, generate_paths()):
                ranking = dict(ranking)
                row = [path] + [ranking[c] for c in identifier.nb_classes]
                writer.writerow(row)
        else:
            for path, (lang, conf) in pool.imap_unordered(cl_path, generate_paths()):
                writer.writerow((path, lang, conf))
    else:
        import sys
        if sys.stdin.isatty():
            # Interactive mode
            while True:
                try:
                    print(">>>", end=' ')
                    text = input()
                except Exception as e:
                    print(e)
                    break
                print(_process(text))
        else:
            # Redirected
            if options.line:
                for line in sys.stdin:
                    print(_process(line))
            else:
                print(_process(sys.stdin.read()))


if __name__ == "__main__":
    main()
