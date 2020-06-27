from query import *
from create import *
import pytest

stemmer = PorterStemmer()

# check read files
def test_read_title():
    result = {'0': '2001 space odyssey', '1': 'armstrong first man moon', '2': 'yuri gagarin first man orbit space',
     '3': 'kubrik movi includ full metal jacket 2001 space odyssey', '4': 'danger killer kiss',
     '5': 'killer murder woman kiss', '6': 'dog killed child', '7': 'full metal jacket bullet killer',
     '8': 'recent nasa program space', '9': 'clockwork orange', '10': 'orange swiss clockwork',
     '11': 'fear destruct glory clockwork', '12': 'homer odyssey clockwork'}
    ranked = [('0', 0.9), ('9', 0.8)]
    assert read_titles('mytitle.dat') == result
    assert match_title(result,ranked) ==[('2001 space odyssey', 0.9),('clockwork orange',0.8)]

def test_read_stopword():
    keys = list(read_stopwords('stopwords.dat').keys())
    assert keys[0] == 'a'

def test_read_stopword2():
    keys = list(read_stopwords2('stopwords.dat').keys())
    assert keys[0] == 'a'

#####################################################
inverted_index = read_inverted_index('myindex.dat')
stopwords = read_stopwords('stopwords.dat')
titles = read_titles('mytitle.dat')
#######################################################

# Test cases for TF-IDF
# Test cases for testing get_tf_matrix & get_doc_score
def test_multi_tfidf():
    assert multiply_tf_idf(tf=5, idf=0.1) == 0.5

def test_tf_idf():
    matched_ids = ['0','2','3','8']
    q_word = ["2001", "space"]
    doc_matrix = [[1, 1], [0, 1], [1, 1], [0, 1]]
    assert get_tf_matrix(matched_ids,inverted_index,q_word) == [[1, 1], [0, 1], [1, 1], [0, 1]]
    assert get_doc_score(doc_matrix, inverted_index, matched_ids, q_word) == \
           [('0', 0.9751424258351847), ('3', 0.9751424258351847), ('2', 0.5328498146926454), ('8', 0.5328498146926454)]

def test_tf_idf2():
    matched_ids = ['0','3']
    q_word = ["2001", "space"]
    doc_matrix = [[1, 1], [1, 1]]
    assert get_tf_matrix(matched_ids, inverted_index, q_word) == doc_matrix
    assert get_doc_score(doc_matrix, inverted_index, matched_ids, q_word) == \
           [('0', 0.9751424258351847), ('3', 0.9751424258351847)]

def test_tf_idf3():
    matched_ids = ['0','2','3','8']
    q_word = ["snow", "space"]
    doc_matrix = [[0, 1], [0, 1], [0, 1], [0, 1]]
    assert get_tf_matrix(matched_ids, inverted_index, q_word) == doc_matrix
    assert get_doc_score(doc_matrix, inverted_index, matched_ids, q_word) == \
           [('0', 1.0), ('2', 1.0), ('3', 1.0), ('8', 1.0)]

# test QueryFactory that find the type of queries
def test_one_word():
    assert type(QueryFactory.create('oneword')).__name__ == 'OneWordQuery'
    assert type(QueryFactory.create('free text')).__name__ == 'FreeTextQuery'
    assert type(QueryFactory.create('"2001 space"')).__name__ == 'PhraseQuery'
    assert type(QueryFactory.create('2001 AND space')).__name__ == 'BooleanQuery'
    assert QueryFactory.create('')== None

# check boolean bad input
def test_check_bool():
    assert check_bool('(hello And') == False
    assert check_bool('(2001 AND space)') == True


# check print function
def test_print_title_score():
    assert print_title([('first', 0.9),('second', 0.5)]) == None
    assert print_ids(['0','5','7']) == None
    assert print_title_score([('first', 0.9),('second', 0.5)]) == None

# New assertions added in Part 3a

def test_read_pagerank():
    output = read_pagerank('score_test.dat')
    assert output == {'2225': '0.15287491341411455', '2226': '0.1257812309341992', '3308': '0.0440846451228826'}

def test_match_pagerank():
    # This test uses the large BreakingBad xml file
    pagerank_index = read_pagerank('score_index.dat')
    pages1 = [2025,2225,3308,26527]
    pages2 =  [1814,2141,11180,26526]
    output1 = pagerank_match(pages1,pagerank_index)
    output2 = pagerank_match(pages2, pagerank_index)
    # output format is a sorted list of tuples [(page_id,score)]
    assert output1 == [('2225',0.15287491341411455), ('3308',0.0440846451228826), ('26527',0.006910511000563958), ('2025',0.006798454060094076)]
    assert output2 == [('1814',0.03619140571533333), ('11180',0.014361153275822595), ('26526',0.010967876502194615), ('2141',0.007285544600645473)]

#################################################
@pytest.fixture
def index():
    index = read_inverted_index('myindex2.dat')
    return index

@pytest.fixture
def stopwords():
    stopwords = read_stopwords('stopwords.dat')
    return stopwords

##################################################
def test_oneword_small(index,stopwords):
    test = '2001'
    query_obj = QueryFactory.create(test)
    query_obj.remove()
    query_obj.lower_q()
    query_obj.obtain_tokens()
    query_obj.filter_tokens(stopwords)
    query_obj.stem()
    match_ids = query_obj.match(index)
    match_ids = [int(x) for x in match_ids]
    match_ids.sort()
    assert match_ids == [0,3,12]
    ranked = query_obj.vector_space(index, match_ids)
    assert ranked ==[('12', 0.5498764007975351), ('0', 0.32585268195409484), ('3', 0.11279515913795593)]

def test_freetext_small(index,stopwords):
    test = 'orange clockwork'
    query_obj = QueryFactory.create(test)
    query_obj.remove()
    query_obj.lower_q()
    query_obj.obtain_tokens()
    query_obj.filter_tokens(stopwords)
    query_obj.stem()
    match_ids = query_obj.match(index)
    match_ids = [int(x) for x in match_ids]
    match_ids.sort()
    assert match_ids == [4,5,9,10,11,12]

def test_phrasequery_small(index,stopwords):
    test = '"2001 odyssey moon"'
    query_obj = QueryFactory.create(test)
    query_obj.remove()
    query_obj.lower_q()
    query_obj.obtain_tokens()
    query_obj.filter_tokens(stopwords)
    query_obj.stem()
    match_ids = query_obj.match(index)
    match_ids = [int(x) for x in match_ids]
    match_ids.sort()
    assert match_ids == [0]

def test_boolean_small(index,stopwords):
    test = 'hello AND find AND nemo'
    query_obj = QueryFactory.create(test)
    match_ids = query_obj.match(index, stopwords)
    query_obj.remove()
    query_obj.lower_q()
    query_obj.obtain_tokens()
    query_obj.filter_tokens(stopwords)
    query_obj.stem()
    match_ids = [int(x) for x in match_ids]
    match_ids.sort()
    assert match_ids == [3]

def test_boolean_OR_small(index,stopwords):
    test = 'hello OR find AND nemo'
    query_obj = QueryFactory.create(test)
    match_ids = query_obj.match(index, stopwords)
    query_obj.remove()
    query_obj.lower_q()
    query_obj.obtain_tokens()
    query_obj.filter_tokens(stopwords)
    query_obj.stem()
    match_ids = [int(x) for x in match_ids]
    match_ids.sort()
    assert match_ids == [0,2,3,7]