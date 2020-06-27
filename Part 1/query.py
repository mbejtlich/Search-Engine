import argparse
import json
import re
import boolparser
from nltk.stem import PorterStemmer

def read_stopwords(stopwords_path):
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

def multiply_tf_idf(tf,idf):
    return tf*idf

def read_titles(titles):
    """ This function reads the title file line by line, returning a dictionary of ids and associated titles
    inputs:
    stopwords_path - Path of titles file as a string
    outputs:
    output - dictionary of stopwords (Key = page id: value = page title)
    """

    f = open(titles,"r")
    content = f.readlines()
    content = [line.strip('\n') for line in content]
    f.close()

    # Create dictionary of stopwords
    output = {}
    for line in content:
        split_line = line.split(None, 1)
        output[split_line[0]]=split_line[1]
    return output

def match_title(titles_dict,ranked):

    output = []
    for doc in ranked:
        id = doc[0]
        score = doc[1]
        title = titles_dict[id]
        output = title + ' ' + score
    return output

def compare(a,b):
    """
    inputs:
    a - list of positions for first term
    b - list of positions for second term
    outputs:
    output - dictionary of match --> {page_id:position}
    """
    output = []
    a_ind = 0
    b_ind = 0
    len_a = len(a)
    len_b = len(b)
    while a_ind <= len(a)-1 and b_ind <= len(b)-1:
        current_a = a[a_ind]
        current_b = b[b_ind]
        if a[a_ind] + 1 > max(b):
            break
        if a[a_ind] + 1 == b[b_ind]:
            output.append(b[b_ind])
            b_ind += 1
            a_ind += 1
        elif a[a_ind] + 1 < b[b_ind]:
            a_ind += 1
        else:
            b_ind += 1
    return output

def read_query_file(query_path):
    """ This function reads each query, line by line, and returns a list of queries as strings
    query_path — path of query file as a string
    outputs:
    content - list of query terms
    """
    f = open(query_path,'r')
    content = f.readlines()
    content = [line.strip('\n') for line in content]
    f.close()
    return content

def read_inverted_index(inverted_index_path):
    """ This function
    inputs:
    inverted_index_path – path of inverted index file as string
    outputs:
    content - {word:[{page_id:[[position,..],tf]},idf]..}
    """
    with open(inverted_index_path,'r') as f:
        content = json.load(f)
    return content

class Query:
    """ Represents a base query. Used by the index to find the right cases.
    """
    def __init__(self,query_string):
        self.query_string = query_string

    def lower_q(self):
        self.query_string = self.query_string.lower()

    def obtain_tokens(self):
        self.query_string = self.query_string.split()

    def filter_tokens(self,stopwords):
        remove_stop = []
        for idx, val in enumerate(self.query_string):
            if val in stopwords:
                pass
            else:
                remove_stop.append(self.query_string[idx])
        self.query_string = remove_stop

    def stem(self):
        ps = PorterStemmer()
        self.query_string = [ps.stem(w) for w in self.query_string]

    def titles(self,rank):
        pass

    # def match(self,index):
    #     """ Returns the document IDs which match the given query
    #     """
    #     raise NotImplemented

class OneWordQuery(Query):
    """ One Word Query subclass
    """
    def __init__(self, query_string):
        super().__init__(query_string)

    def remove(self):
        pass

    def match(self,inverted_index):
        """
        inputs:
        index – {word:{page_id:[position,..]}..}
        outputs:
        content - return a sorted list of matching document ids
        """
        output = []
        x = self.query_string
        x = self.query_string[0]
        #y = inverted_index['2001']

        if self.query_string[0] in inverted_index:
            inner = inverted_index[self.query_string[0]][0]
            for key in inner:
                output.append(key)
        else:
            output.append(' ')
        return output

    def vector_space(self,inverted_index,ids):
        query = self.query_string[0]
        ids = list(map(str, ids))
        output = []
        for id in ids:
            idf = inverted_index[query][0][id][1]
            tf = inverted_index[query][1]
            score = multiply_tf_idf(tf, idf)
            output.append((id, score))
        sorted_by_score = sorted(output, key=lambda tup: tup[1], reverse=True)
        return sorted_by_score

class FreeTextQuery(Query):
    """ Free Text Query subclass
    """
    def __init__(self, query_string):
        super().__init__(query_string)

    def remove(self):
        pass

    def match(self,inverted_index):
        """
        inputs:
        self-query_string - list of tokens
        index – {word:{page_id:[position,..]}..}
        outputs:
        content - return a sorted list of matching document ids
        """
        total = []
        #print(type(self.query_string))
        for query in self.query_string:
            if query in inverted_index:
                inner = inverted_index[query][0]
                for key in inner:
                    #print(key)
                    total.append(key)

        intersection = set(total)
        output = list(intersection)

        # If no matches, print empty string
        if output == []:
            output.append(' ')
        return output

class PhraseQuery(Query):
    """ Phrase Query subclass
    """
    def __init__(self, query_string):
        super().__init__(query_string)

    def remove(self):
        #print(self.query_string)
        p = re.compile('"')
        self.query_string=p.sub('',self.query_string)

    def match(self,inverted_index):
        """
        inputs:
        self-query_string - list of tokens
        index – {word:{page_id:[position,..]}..}
        outputs:
        content - return a sorted list of matching document ids
        """
        total = []
        # Extract position lists
        for query in self.query_string:
            x = [key for key in inverted_index[query][0].keys()]
            total.append(set(x))

        # Get intersecting positions
        page_ids = set.intersection(*total)
        page_ids = list(page_ids)
        count = 0
        output = []
        dict_temp = {}
        for w in range(len(self.query_string)):
            if count == len(self.query_string)-1:
                if dict_temp == {}:
                    output.append(' ')
                else:
                    output.extend(list(dict_temp.keys()))
                break
            token_1 = self.query_string[w]
            token_2 = self.query_string[w+1]
            #print(token_1)
            #print(token_2)

            if w >= 1:
                for p in page_ids:
                    x = dict_temp
                    if p in dict_temp:
                        token_1_position = dict_temp[p]
                        token_2_position = inverted_index[token_2][0][p][0]
                        token_1_position.sort()
                        token_2_position.sort()
                        result = compare(token_1_position,token_2_position)
                        if result != []:
                            dict_temp[p]=result
                        else:
                            del dict_temp[p]
            if w == 0:
                for p in page_ids:
                    token_1_position = inverted_index[token_1][0][p][0]
                    token_2_position= inverted_index[token_2][0][p][0]
                    #print(token_1_position)
                    #print(token_2_position)
                    token_1_position.sort()
                    token_2_position.sort()
                    result = compare(token_1_position,token_2_position)
                    if result != []:
                        dict_temp[p] = result
            count += 1
        return output

class BooleanQuery(Query):
    """ Boolean Query subclass
    """
    def __init__(self, query_string):
        super().__init__(query_string)

    def match(self,inverted_index,stopwords):

        ast = boolparser.bool_expr_ast(self.query_string)
        #print(ast)
        output= rec(ast,inverted_index,stopwords)
        return output

def rec(ast,inverted_index,stopwords):

    if isinstance(ast,str):
        # Get id of string (return id)
        query_obj = OneWordQuery(ast)
        query_obj.remove()
        query_obj.lower_q()
        query_obj.obtain_tokens()
        query_obj.filter_tokens(stopwords)
        query_obj.stem()
        match_ids = query_obj.match(inverted_index)
        return match_ids

    if isinstance(ast,tuple):
        if ast[0] == "AND":
            # do processing
            total_list = []
            for element in ast[1]:
                single_list = rec(element,inverted_index,stopwords)
                total_list.append(set(single_list))
            page_ids = set.intersection(*total_list)
            page_ids = list(page_ids)
            return page_ids
            # return union
        else:
            # ast[0] == 'OR'
            total_list = []
            for element in ast[1]:
                single_list = rec(element, inverted_index, stopwords)
                total_list.append(set(single_list))
            page_ids = set.union(*total_list)
            page_ids = list(page_ids)
            return page_ids

class QueryFactory:
    """ This class looks reads in the queries as a string and determines the appropriate class for each query
    inputs:
    outputs:
    """
    @staticmethod
    def create(query_string:str):
        if not query_string:
            return None
        if list(query_string[0]) == ['"'] and list(query_string[-1])==['"']:
            return PhraseQuery(query_string)
        if any([t in query_string for t in ('(',')','AND','OR')]):
            return BooleanQuery(query_string)
        words = query_string.split()
        if len(words) == 1:
            return OneWordQuery(query_string)
        if len(words) > 1:
            return FreeTextQuery(query_string)
        else:
            raise ValueError('This query string is not a valid type.')

def main():
    #Create ArgumentParser object
    #parser = argparse.ArgumentParser(description='Query parser')

    #Add input and output filepaths
    # parser.add_argument('my_stopwords', help='Path to stopwords dat file')
    # parser.add_argument('my_inverted_index', help='Path to inverted index .dat file')
    # parser.add_argument('my_titles', help='Path to titles dat file')
    # parser.add_argument('my_queries', help='Path to queries dat file')
    # args = parser.parse_args()

    # print("~ Stopwords path: {}".format(args.my_stopwords))
    # print("~ Inverted index path: {}".format(args.my_inverted_index))
    # print("~ Titles path: {}".format(args.my_titles))
    # print("~ Queries path: {}".format(args.my_queries))


    # stopwords = read_stopwords(args.my_stopwords)
    # query = read_query_file(args.my_queries)
    # #print(query)
    # inverted_index = read_inverted_index(args.my_inverted_index)

    stopwords = read_stopwords('stopwords.dat')
    query = read_query_file('small_q_boolean.dat')
    # print(query)
    inverted_index = read_inverted_index('index_test.dat')
    titles_dict = read_titles('myTitles.dat')

    for q in query:
        query_obj = QueryFactory.create(q)
        if type(query_obj).__name__ == 'BooleanQuery':
            initial = query_obj.query_string
            match_ids = query_obj.match(inverted_index,stopwords)
            match_ids = set(match_ids)
            match_ids = [int(x) for x in match_ids]
            match_ids.sort()
            #print(initial)
            print(*match_ids)
        else:
            if query_obj == None:
                #print('Empty Query')
                print(' ')
            else:
                initial = query_obj.query_string
                query_obj.remove()
                query_obj.lower_q()
                query_obj.obtain_tokens()
                query_obj.filter_tokens(stopwords)
                query_obj.stem()
                match_ids = query_obj.match(inverted_index)
                query = query_obj.query_string
                if match_ids[0] == ' ':
                    print(' ')
                else:
                    ranked = query_obj.vector_space(inverted_index,match_ids)
                    titles_score = match_title(titles_dict,ranked)
                    print(*titles_score, sep='\n')
                    #match_ids = set(match_ids)
                    #match_ids = [int(x) for x in match_ids]
                    #match_ids.sort()
                    #print(initial)
                    #print(stem_result)
                    #print(*match_ids)

if __name__ == '__main__':
    main()