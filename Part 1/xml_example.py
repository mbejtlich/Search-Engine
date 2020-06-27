import xml.etree.ElementTree as ET
import re

#file_name = 'small.xml'
#full_file = os.path.abspath(os.path.join('test',file_name))

tree = ET.parse('small.xml')
root = tree.getroot()
ns = re.match('{.*}',root.tag)

if ns == None:
    ns = ''
else:
    ns=ns.group()

pages = root.findall(ns+'page')

for p in pages:
    page_title = p.find(ns+'title').text
    page_id = p.find(ns+'id').text
    for t in p.iter(ns+'text'):
        page_text = t.text
    print('* {}|{}|{}'.format(page_title,page_id,page_text))
    # full_text = page_title + ' ' + page_text
    # full_text.lower()
    # #print(full_text)
    #page = (page_id,page_title,[full_text])


# ids = root.findall(ns+'page/'+ns+'id')
# for id in ids:
#     print(id.text)

# titles = root.findall(ns+'page/'+ns+'title')
# for title in titles:
#     print(title.text)

# pages = root.findall(ns+'page/')
# for page in pages:
#     print(page.find(ns+'title').text)

# titles = root.findall('page/title')
# for t in titles:
#     print(t.text)
#
# ids = root.findall('page/id')
# for id in ids:
#     print(id.text)




