# query_string = '"hello we are"'
#
# if query_string[0] == '"' and query_string[-1] == '"':
#     x = 2
# else:
#     x = 1
#
# print(x)
# print(query_string[-

# test = {'m1':{'23':None,'24':None,'25':None}}
#
# p=test['m1'].keys()
# print(p)
#
# x = [key for key in test['m1'].keys()]
# print(x)

# s1 = set(['22','23','45'])
# s2 = set(['22','23'])
# s3 = set(['2','8','23','25'])
# test = [s1,s2,s3]
# u = set.union(*test)
# print(u)

a = [1,3,2,4]
a.sort()

print(a)

# x = ['hello','dog','cat']
#
# count = 0
# for w in range(len(x)):
#     print(w)
#     if count == len(x)-1:
#         break
#     token_count = len(x)
#     y = w + 1
#     word1 = x[w]
#     word2 = x[y]
#     count += 1

# def compare(a,b):
#     """
#     inputs:
#     a - list of positions for first term
#     b - list of positions for second term
#     outputs:
#     output - dictionary of match --> {page_id:position}
#     """
#     output = []
#     b_ind = 0
#     for x in range(len(a)):
#         temp = a[x] + 1
#         if temp == b[b_ind]:
#             output.append(b[b_ind])
#             b_ind += 1
#     return output
#
# a = [1,2,3,4]
# b = [2,4,5,7,8]
#
# x=compare(a,b)
# print(x)

# match_ids = ' '
# match_ids = set(match_ids)
# match_ids = [int(x) for x in match_ids]
# match_ids.sort()
# print(*match_ids)

def compare(a,b):
    """
    inputs:
    a - list of positions for first term
    b - list of positions for second term
    outputs:
    output - dictionary of match --> {page_id:position}
    """
    output = []
    b_ind = 0
    for x in range(len(a)):
        temp = a[x] + 1
        if temp >= max(b):
            break
        if temp == b[b_ind]:
            output.append(b[b_ind])
            b_ind += 1
    return output

a = [30]
b = [1,2,4,6,29]

x = compare(a,b)
print(x)