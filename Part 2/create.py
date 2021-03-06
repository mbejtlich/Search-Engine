import argparse
import math
import xml.etree.ElementTree as ET
import re
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()


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
    #print(output)
    return output

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

    output_file = open(title_index_path, "w")
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
    with open(inverted_index_path, 'w') as outfile:
       json.dump(dict_word, outfile, indent=4)

def main():
    try:
        # Create ArgumentParser object
        parser = argparse.ArgumentParser(description='Create inverted index and title files')

        #Add input and output filepaths
        parser.add_argument('my_stopwords', help='Path to stopwords dat file')
        parser.add_argument('my_collection', help='Path to collection xml file')
        parser.add_argument('my_index', help='Path to index dat file')
        parser.add_argument('my_titles', help='Path to titles dat file')
        args = parser.parse_args()
        #
        print("~ Stopwords path: {}".format(args.my_stopwords))
        print("~ Collection path: {}".format(args.my_collection))
        print("~ Index path: {}".format(args.my_index))
        print("~ Titles path: {}".format(args.my_titles))

        stopwords=read_stopwords2(args.my_stopwords)
        t1=read_collection(args.my_collection,stopwords,args.my_titles)
        create_invertedindex(t1,args.my_index)

        # stopwords=read_stopwords('stopwords.dat')
        # collection=read_collection('pixar_pages_current .xml',stopwords,'myTitlesPixar2.dat')
        # create_invertedindex(collection,'myIndexPixar2.dat')
    except SyntaxError:
        print('ERROR:BAD XML File encountered. Exiting')

if __name__ == '__main__':
    main()