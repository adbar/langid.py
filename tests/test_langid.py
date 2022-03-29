
import subprocess
import sys

from io import StringIO
from pathlib import Path

import py3langid as langid
from py3langid.langid import LanguageIdentifier, MODEL_FILE


def test_langid():
    '''Test if the language detection functions work'''
    # basic classification
    text = b'This text is in English.'
    assert langid.classify(text)[0] == 'en'
    assert langid.rank(text)[0][0] == 'en'
    text = 'This text is in English.'
    assert langid.classify(text)[0] == 'en'
    assert langid.rank(text)[0][0] == 'en'
    text = 'Test Unicode sur du texte en français'
    assert langid.classify(text)[0] == 'fr'
    assert langid.rank(text)[0][0] == 'fr'
    # normalization of probabilities
    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    _, prob = identifier.classify(text)
    assert 0 <= prob <= 1
    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=False)
    _, prob = identifier.classify(text)
    assert prob < 0
    # subset of target languages
    identifier.set_languages(['de', 'en', 'fr'])
    assert identifier.classify('这样不好')[0] != 'zh'



def test_redirection():
    '''Test if STDIN redirection works'''
    thisdir = Path(__file__).parent
    langid_path = str(thisdir.parent / 'py3langid' / 'langid.py')
    readme_path = str(thisdir.parent / 'README.rst')
    with open(readme_path, 'rb') as f:
        readme = f.read()
    result = subprocess.check_output(['python3', langid_path, '-n'], input=readme)
    assert b'en' in result and b'1.0' in result



def test_cli():
    '''Test console scripts entry point'''
    result = subprocess.check_output(['langid', '-n'], input=b'This should be enough text.')
    assert b'en' in result and b'1.0' in result
    result = subprocess.check_output(['langid', '-n', '-l', 'bg,en,uk'], input=b'This should be enough text.')
    assert b'en' in result and b'1.0' in result
