import sqlcl

f = open('query.txt', 'r')
query = f.read()

lines = sqlcl.query(query).readlines()
print lines[0]
