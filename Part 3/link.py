import numpy as np
import json
import re
from argparse import ArgumentParser
from typing import Dict, List
from xml.etree import cElementTree


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

def main(collection_path: str):
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

    with open('scores.dat', 'w') as fp:
        for id_, score in sorted(list(scores.items()), key=lambda p: -p[1]):
            fp.write('{}|{}\n'.format(id_, score))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('collection')
    args = parser.parse_args()
    main(args.collection)
