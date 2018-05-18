import os
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]


cats=['alt.atheism', 'talk.religion.misc',
             'comp.graphics', 'sci.space']

newsgroup_train=fetch_20newsgroups(subset='test',categories=cats)
#print(len(newsgroup_train.data))

path = "./data"
files= os.listdir(path)
s = []
for file in files:
     if not os.path.isdir(file):
          f = open(path+"/"+file);
          iter_f = iter(f);
          str = ""
          for line in iter_f:
              str=str+line
          s.append(str)
#print(s)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)

#use this for preset data
#X = vectorizer.fit_transform(corpus)

#use this for 20newsgroup
#X = vectorizer.fit_transform(newsgroup_train.data)


#use this for test news
X = vectorizer.fit_transform(s)



word = vectorizer.get_feature_names()
weight = X.toarray()#.tolist()
#this is function for Cosine Similarity
#dist=1-cosine_similarity(X)

#this is also for the distance function I wrote
lenth=len(weight[0])
for i in range(lenth):
    if i>=lenth:
        break
    we=weight[0][i]
    for j in range(len(weight)):
        if weight[j][i]>we:
            we=weight[j][i]
    if we<0.01:
        for j in range(len(weight)):
            del weight[j][i]
        del word[i]
        lenth-=1
print(word)
print(weight)
print(len(word))
#print(dist)

#this is distance function I wrote
def getdist(a, b):
    sumSquares = 0
    if a == b:
        return 0
    for i in range(len(weight[0])):
        sumSquares += math.pow(weight[a][i] - weight[b][i], 2)
    return math.sqrt(sumSquares)
#print(getdist(0, 3))


def findmin(M):
    min = 1000
    x = 0
    y = 0
    if len(M) > 1:
        for i in range(len(M)):
            for j in range(i):
                if M[i][j] < min:
                    min = M[i][j]
                    x = i
                    y = j
    return x, y, min


def AGNES(thrshld1, thrshld2):
    M = []
    for i in range(len(weight)):
        Mi = []
        for j in range(i):
            #use this for distance function I write but it doesn't work well with large dataset
            Mi.append(getdist(i, j))

            #use this for Euclidean Distance
            #Mi.append(numpy.linalg.norm(weight[j]-weight[i]))

            #use this for Cosine Similarity
            #Mi.append(dist[i][j])
        M.append(Mi)
    C = []
    for i in range(len(weight)):
        Ci = [i]
        C.append(Ci)
    print('\n Distance matrix')
    for i in M:
        print(i)
    while True:
        x, y, minw = findmin(M)
        if minw > thrshld1:
            break
        Ci = C[x]
        C[y].extend(Ci)
        for i in range(len(M[y])):
            if i != x and i != y:
                M[y][i] = (M[x][i]*len(C[x]) + M[y][i]*len(C[y])) / (len(C[x])+len(C[y]))
        del C[x]
        print('\n Groups:')
        for i in C:
            print(i)
        del M[x]
        for i in range(x,len(M)):
            del M[i][x]
        if len(M)<=thrshld2:
            break
    return C

result=AGNES(1000, 4)
count=0
for i in result:
    count+=len(i)
print(count)
