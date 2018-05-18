from numpy import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(power(int(vector2) - int(vector1), 2)))  # 求这两个矩阵的距离，vector1、2均为矩阵


def kmeans(centroids,dataSet):
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)
    return centroids, clusterAssment

def load_stop_words(stop_word_file):
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words


def load_test_word(test_word_file):
    test_words = []
    for line in open(test_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                test_words.append(word)
    return test_words


def Prediction(test):
    test1 = ' '.join(word for word in test)
    Z = []
    Z.append(test1)

    P = vectorizer.transform(Z)
    prediction = km.predict(P)
    print("The input data shoule be clustered into cluster")
    print(prediction)
    return prediction

def Print(order_centroids,true_k):
    terms = vectorizer.get_feature_names()
    data_samples = dataset.data[:10]
    dtm_vectorizer = CountVectorizer()
    dtm = dtm_vectorizer.fit_transform(data_samples)
    m = [0, 0, 0, 0]
    print("ten elements example from clustering")
    for i in range(true_k):
        print("clustering  %d:" % i, end=' ')
        for ind in order_centroids[i, :10]:
            if (dtm_vectorizer.vocabulary_.get(terms[ind]) != None):
                print('%s' % dtm_vectorizer.vocabulary_.get(terms[ind]), end=' ')
                m[i] += 1
        print()
    print()
    print(m)


if __name__ == "__main__":
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=False, random_state=42,
                                 remove=('headers', 'footers', 'quotes'))
    labels = dataset.target

    stopPath = "SmartStoplist.txt"
    Y = load_stop_words(stopPath)
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=Y)
    labels = dataset.target
    k = 3
    X = vectorizer.fit_transform(dataset.data)
    print("n_samples: %d, n_features: %d" % X.shape)
    numSamples = X.shape[0]
    dim = X.shape[1]
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)
    centroids = km.cluster_centers_.argsort()[:, ::-1]
    #kmeans(centroids,labels)
    Print(centroids,k)

    #test prediction
    testPath = "test.txt"
    test = load_test_word(testPath)
    Prediction(test)


    #evaluators
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))
