import json

# index = {"2001": [{"0": [[0,2],2],"3": [[6],1],"12": [[5,6,7],3]},1],
#          "space": [{"0": [[1],1],"2": [[5],1],"3": [[7],1],"8": [[3,5],2]},1],
#          "odyssey": [{"0": [[3],1],"3": [[8],1],"12": [[1],1]},1],
#          "moon": [{"0": [[4],1],"1":[[3],1]},1]
#         }
# with open('index_test.dat', 'w') as outfile:
#     json.dump(index, outfile, indent=4)

with open('index_test.dat', 'r') as f:
    content = json.load(f)
print(content)
test = content['2001'][0]['0'][1]
idf = content['2001'][1]

print(test)
print(idf)

# inner = content['2001'][0]
# for key in inner:
#     print(key)
#
# test = [2,3,4,5,2,4,6]
# test2 = set(test)
# print(test2)