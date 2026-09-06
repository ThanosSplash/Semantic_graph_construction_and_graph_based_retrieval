
import data_loading as dt
import pytest
def test_empty_vals_dev():
    """Testing if saved texts are nan or empties on dev set"""
    query_text, _, corpus_text = dt.load_texts()
    empty_query_vals = []
    for id in query_text.keys():
        if query_text[id] == "nan" or query_text[id] == "" or query_text[id] == " ":
            empty_query_vals.append(id)

    assert empty_query_vals == []
    # Testing for corpus
    empty_corpus_vals = []
    for id in corpus_text.keys():
        if corpus_text[id] == "nan" or corpus_text[id] == "" or corpus_text[id] == " ":
            empty_corpus_vals.append(id)


    assert empty_corpus_vals == []


def test_empty_vals_in_test_set():
    """Testing if saved texts are nan or empties on test set"""
    query_text, _ = dt.load_texts_tests()
    empty_query_vals = []
    for id in query_text.keys():
        if query_text[id] == "nan" or query_text[id] == "" or query_text[id] == " ":
            empty_query_vals.append(id)
    assert empty_query_vals == []


def test_text_emb_same_ids_dev():
    """Testing if dictionary for embeddings and dictionary for texts have the same ids for dev split"""
    query_text, _, corpus_text = dt.load_texts()
    query, _, corpus = dt.load_data()
    assert all(id in query_text for id in query.keys())

def test_text_emb_same_ids_test():
    """Testing if dictionary for embeddings and dictionary for texts have the same ids for test split"""
    query_text, _ = dt.load_texts_tests()
    query, _ = dt.load_data_tests()
    assert all(id in query_text for id in query.keys())

def test_duplicates_dev_test_text_splits():
    """Testing if duplicates exists between dev and test splits for text"""
    query_text_test, answer_text_test = dt.load_texts_tests()
    query_text, answer_text, corpus_text = dt.load_texts()
    duplicates = []
    for key in query_text_test.keys():
        if key in query_text:
            duplicates.append(key)

    assert duplicates == []

def test_duplicates_dev_test_splits():
    """Testing if duplicates exists between dev and test splits for dev"""
    query_test, answer_test = dt.load_data_tests()
    query, answer, corpus = dt.load_data()
    duplicates = []
    for key in query_test.keys():
        if key in query:
            duplicates.append(key)
    assert duplicates == []

def test_dev_split():
    """Testing if small, medium, long splits have valid ids for dev"""
    small_dev, medium_dev, long_dev = dt.load_splits_dev()
    all_dev = small_dev + medium_dev + long_dev

    query_text, answer_text, corpus_text = dt.load_texts()
    query, answer, corpus = dt.load_data()

    assert all(id in query_text for id in all_dev)
    assert all(id in query for id in all_dev)


def test_testing_split():
    """Testing if small, medium, long splits have valid ids for test"""
    small_test, medium_test, long_test = dt.load_splits_test()
    all_test = small_test + medium_test + long_test
    query_text, answer_text = dt.load_texts_tests()
    query, answer = dt.load_data_tests()

    assert all(id in query_text for id in all_test)
    assert all(id in query for id in all_test)

