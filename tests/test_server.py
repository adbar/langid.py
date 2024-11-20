import json

from unittest.mock import MagicMock

import pytest

from py3langid.langid import application


@pytest.fixture
def mock_start_response():
    return MagicMock()

def test_detect_put(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'PUT',
        'CONTENT_LENGTH': 10,
        'wsgi.input': MagicMock(read=lambda x: b'This is a test'),
        'PATH_INFO': '/detect'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '200 OK'
    assert json.loads(response[0].decode('utf-8'))['responseData']['language'] == 'en'

def test_detect_get(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'GET',
        'QUERY_STRING': 'q=This+is+a+test',
        'PATH_INFO': '/detect'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '200 OK'
    assert json.loads(response[0].decode('utf-8'))['responseData']['language'] == 'en'

def test_detect_post(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'POST',
        'CONTENT_LENGTH': 10,
        'wsgi.input': MagicMock(read=lambda x: b'q=Hello+World'),
        'PATH_INFO': '/detect'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '200 OK'
    assert json.loads(response[0].decode('utf-8'))['responseData']['language'] == 'en'

def test_rank_put(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'PUT',
        'CONTENT_LENGTH': 10,
        'wsgi.input': MagicMock(read=lambda x: b'Hello World'),
        'PATH_INFO': '/rank'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '200 OK'
    assert json.loads(response[0].decode('utf-8'))['responseData'] is not None

def test_rank_get(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'GET',
        'QUERY_STRING': 'q=Hello+World',
        'PATH_INFO': '/rank'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '200 OK'
    assert json.loads(response[0].decode('utf-8'))['responseData'] is not None

def test_rank_post(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'POST',
        'CONTENT_LENGTH': 10,
        'wsgi.input': MagicMock(read=lambda x: b'q=Hello+World'),
        'PATH_INFO': '/rank'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '200 OK'
    assert json.loads(response[0].decode('utf-8'))['responseData'] is not None

def test_invalid_method(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'DELETE',
        'PATH_INFO': '/detect'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '405 Method Not Allowed'

def test_invalid_path(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'GET',
        'PATH_INFO': '/invalid'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '404 Not Found'

def test_empty_path(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'GET',
        'PATH_INFO': ''
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '404 Not Found'

def test_no_query_string(mock_start_response):
    environ = {
        'REQUEST_METHOD': 'GET',
        'PATH_INFO': '/detect'
    }
    response = application(environ, mock_start_response)
    assert mock_start_response.call_args[0][0] == '400 Unknown Status'
    assert json.loads(response[0].decode('utf-8'))['responseData'] is None
