#################### Import Packages ############################################
import numpy as np
import math
import re
import json
from typing import Dict, List
import argparse
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from xml.etree import cElementTree
from nltk.stem import PorterStemmer
import os
from nltk.tokenize import sent_tokenize, word_tokenize
###################### Make a directory #####################################
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
###################### Read stop words ########################################
def read_stopwords2(stopwords_path):
    """ This function reads the stopwords file line by line, returning a list of the stopwords
    inputs:
    stopwords_path - Path of stopwords file as a string
    outputs:
    output - dictionary of stopwords (Key = word: value = None)
    """
    output = {}
    f = open(stopwords_path,"r")
    content = f.readlines()
    content = [line.strip('\n') for line in content]
    f.close()
    # Create dictionary of stopwords
    for word in content:
        output[word] = None
    return output
###################### Read Collection #########################################
def read_collection(collection_path,stopwords_dict,title_index_path):
    """ This function reads the collection line by line and converts it to a list of tuples, containing page id,
    title, and a stream of text
    inputs:
    dictionary_path - Path of dictionary as a string
    outputs:
    content - [(page_id, title, [steam of words]), ...]
    """
    ps = PorterStemmer()
    tree = ET.parse(collection_path)
    root = tree.getroot()
    ns = re.match('{.*}', root.tag)
    if ns == None:
        ns = ''
    else:
        ns = ns.group()
    pages = root.findall(ns + 'page')

    output_file = open(os.path.join("index",title_index_path),"w")
    output=[]
    for p in pages:
        page_title = p.find(ns + 'title').text
        page_id = p.find(ns + 'id').text
        output_file.write(page_id + ' ' + page_title)
        output_file.write("\n")
        for t in p.iter(ns + 'text'):
            page_text = t.text
        if page_title == None:
            page_title = ''
        if page_text == None:
            page_text = ''

        full_text = page_title + ' ' + page_text
        full_text=full_text.lower()
        text_split = re.split('[^a-z0-9]*',full_text)
        text_split=list(filter(None,text_split))

        remove_stop = []
        for idx, val in enumerate(text_split):
            if val in stopwords_dict:
                pass
            else:
                remove_stop.append(text_split[idx])
        final_string = [ps.stem(w) for w in remove_stop]
        output.append((page_id,page_title,final_string))
    output_file.close()
    return output
###################### Create Inverted Index with tf-idf #####################################
def create_invertedindex(corpus,inverted_index_path):
    """ This function reads in a list of tuples containing page id, title, and text of the complete corpus and
    generates an inverted index as a JSON file
    corpus - [(page_id, title, [steam of words]), ...]
    output:
    JSON file - {word:{page_id:[[position,..],tf,tf_norm],idf}..}
    """
    dict_word = {}
    total_doc = len(corpus)
    tf = 0
    idf = 0
    tf_norm = 0
    for page in corpus:
        for idx,word in enumerate(page[2]):
            if word in dict_word:
                if page[0] in dict_word[word][0]:
                    doc_id_list = dict_word[word][0][page[0]][0]
                    doc_id_list.append(idx)
                else:
                    #dict_word[word] gets into "[{page[0]:[idx]},idf]",
                    # accessing the 1st element get the dictionary "{page[0]:[[idx],tf]}"
                    # using "[page[0]]" to check the key in the previous dictionary
                    dict_word[word][0][page[0]]=[[idx],tf,tf_norm]
            else:
                dict_word[word]=[{page[0]:[[idx],tf,tf_norm]},idf]
            # Update the tf
            tf = len(dict_word[word][0][page[0]][0])
            dict_word[word][0][page[0]][1] = tf
            # Update normalized tf
            tf_norm = tf/len(page[2])
            dict_word[word][0][page[0]][2] = tf_norm
            # Update the idf
            idf = math.log(float(len(dict_word[word][0]))/total_doc)
            idf = -idf
            dict_word[word][1] = idf

    #print(dict_word.keys())
    with open(os.path.join("index",inverted_index_path), 'w') as outfile:
       json.dump(dict_word, outfile, indent=4)
###########################################################
###########################################################
###########################################################
def parse(collection_path: str) -> Dict[int, List[int]]:
    """Parses the collection file and returns a dictionary mapping
    documents to the documents that they link to.

    The dictionary keys and values are int document ids.

    Note: We recommend that you don't change this code.
    """
    root = cElementTree.parse(collection_path).getroot()
    match = re.match(r'{.*}', root.tag)
    namespace = match.group() if match else ''

    doc_ids = {}
    outlink_titles = {}
    for page in root.iter(namespace + 'page'):
        id_ = int(page.find(namespace + 'id').text)
        title = page.find(namespace + 'title').text
        assert id_ is not None and title is not None
        # Note this doesn't work on the small index, we aren't using
        # the small index anymore in the course
        text = page.find(namespace + 'revision').find(namespace + 'text').text
        if text is None:
            links = []
        else:
            links = extract_links(text)

        doc_ids[title] = id_
        outlink_titles[id_] = links

    outlink_ids = {}
    for id_, titles in outlink_titles.items():
        outlink_ids[id_] = [doc_ids[title]
                            for title in titles
                            if title in doc_ids]

    for id_ in get_isolates(outlink_ids):
        outlink_ids.pop(id_)

    return outlink_ids


def extract_links(text: str) -> List[str]:
    """Returns the links in the body text. The links are
    title strings.

    Note: We recommend that you don't change this code.
    """
    return re.findall(r'\[\[([^\]|#]+)', text)


def get_isolates(outlinks: Dict[int, List[int]]) -> List[int]:
    """Returns all doc ids which have no inbound nor
    outbound links.

    Note: We recommend that you don't change this code.
    """
    connected_ids = set()
    for id_, linked_ids in outlinks.items():
        if linked_ids:
            connected_ids.add(id_)
            connected_ids.update(linked_ids)

    return [id_ for id_ in outlinks if id_ not in connected_ids]

def length_vec(vec):
    # return the length of a vector
    result = np.sqrt(np.dot(vec,vec))
    return result

def rank(outlinks: Dict[int, List[int]],
         eps: float = 0.01,
         d: float = 0.85) -> Dict[int, float]:
    """Returns the PageRank scores of the documents stored in
    outlinks.

    :param outlinks Mapping of doc ids to the ids that they link to
    :param eps The convergence threshold
    :param d The damping factor
    """
    # TODO: Implement PageRank here
    # Damping factor
    eps = 0.01
    d = 0.85
    # create a dictionary that stores index of the pages- {page_id: index}
    index={}
    i = 0
    for key,val in outlinks.items():
        index[key] = i
        i += 1
    # construct transition matrix M with dim(n ,n)
    n = len(outlinks)
    M = np.zeros(shape=(n,n))
    i = 0
    for key, values in outlinks.items():
        if values ==[]:
            M[:,i] = 1/n
        else:
            for item in list(set(values)):
                M[index[item],i] = values.count(item)/len(values)
        i += 1
    # create the constant term that is (1-d)/n, dimension is nx1
    constant = np.empty(n)
    constant.fill((1-d)/n)
    # initialize the result probability vector
    base_case = np.empty(n)
    base_case.fill(1/n)
    PR = np.add(constant,np.multiply( d, M.dot(base_case)))
    compare = length_vec(np.subtract(PR, base_case))
    # Looping to update the probability vector until the previous vector & current vector differ less than eps
    while (compare >= eps):
        PR_new = np.add(constant,np.multiply(d, M.dot(PR)))
        compare = length_vec(np.subtract(PR_new,PR))
        PR = PR_new
    # Create a dictionary that stores the page_id and scores
    rank_dict = {}
    i = 0
    for key,val in outlinks.items():
        rank_dict[key] = PR[i]
        i +=1
    #x = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)
    #print('sort dict', x[0:5])
    return rank_dict

def main_func(collection_path: str):
    """Saves the outlinks dictionary as a JSON file then computes
    and saves the PageRank scores.

    Note: We recommend that you don't change this code.
    """
    try:
        with open('links.json', 'r') as fp:
            with_str_keys = json.load(fp)
            outlinks = {int(key): val for key, val in with_str_keys.items()}
        print('Using existing links file')
    except FileNotFoundError:
        print('Creating new links file')
        outlinks = parse(collection_path)
        with open('links.json', 'w') as fp:
            json.dump(outlinks, fp)

    scores = rank(outlinks)

    #with open('scores.dat', 'w') as fp:
    with open(os.path.join("index", 'score.dat'), 'w') as fp:
        for id_, score in sorted(list(scores.items()), key=lambda p: -p[1]):
            fp.write('{}|{}\n'.format(id_, score))
###########################################################
###########################################################
###########################################################

def main():
    try:
        # Create ArgumentParser object
        parser = argparse.ArgumentParser(description='Create inverted index and title files')

        #Add input and output filepaths
        parser.add_argument('my_stopwords',nargs='?', help='Path to stopwords dat file')
        parser.add_argument('my_collection',nargs='?', help='Path to collection xml file')
        parser.add_argument('index_folder',nargs='?', help='Path to index dat file')
        # parser.add_argument('my_titles', help='Path to titles dat file')
        args = parser.parse_args()
        #
        print("~ Stopwords path: {}".format(args.my_stopwords))
        print("~ Collection path: {}".format(args.my_collection))
        print("~ Index folder".format(args.index_folder))
        # print("~ Index path: {}".format(args.my_index))
        # print("~ Titles path: {}".format(args.my_titles))
        #print(len(vars(args)))

        index = 'myIndex.dat'
        titles = 'myTitles.dat'

        make_dir(args.index_folder)
        stopwords=read_stopwords2(args.my_stopwords)
        t1=read_collection(args.my_collection,stopwords,titles)
        create_invertedindex(t1,index)
        main_func(args.my_collection)

        # stopwords=read_stopwords2('stopwords.dat')
        # collection=read_collection('pixar_pages_current .xml',stopwords,'myTitlesPixar2.dat')
        # create_invertedindex(collection,'myIndexPixar2.dat')
    except SyntaxError:
        print('ERROR:BAD XML File encountered. Exiting')
    except Exception as e:
        print(e)
        print('ERROR: INPUT NUMBER OF ARGUMENTS INCORRECT')

if __name__ == '__main__':
    main()