import numpy as np
# import json
#
# with open('index_test.dat', 'r') as f:
#     content = json.load(f)
# print(content)
#
# # Single word query
# ids = [0,3,12]
# ids = list(map(str,ids))
#
# def multiply_tf_idf(tf,idf):
#     return tf*idf
#
# query = '2001'
# output = []
# for id in ids:
#     idf = content[query][0][id][1]
#     tf = content[query][1]
#     score = multiply_tf_idf(tf,idf)
#     output.append((id,score))
# sorted_by_second = sorted(output, key=lambda tup: tup[1],reverse=True)
#
#
# print(sorted_by_second)

x = '0 2001 Space 2001 odyssey moon'
split_x = x.split(None,1)
print(split_x)