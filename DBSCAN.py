from sklearn.datasets import fetch_20newsgroups
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 
import heapq
from itertools import cycle

from openpyxl import load_workbook
import os
import math

UNCLASSIFIED = False
NOISE = 0

class D_clustering:
	def __init__(self):
		print("Initialized!")

	def allfiles(self):
		# cats = ['alt.atheism', 'rec.autos', 'sci.electronics']
		# newsgroup_train = fetch_20newsgroups(subset='train', categories=cats).data[0:200]
		# all_file = []
		# for file in newsgroup_train:
		# 	newfile = file.encode("utf-8")
		# 	all_file.append(newfile)
		#print("All files extracted!")

		all_file = []
		workbook = load_workbook('projectsample.xlsx')
		sheets = workbook.get_sheet_names()
		booksheet  = workbook.get_sheet_by_name(sheets[1])

		rows = booksheet.rows
		count = 0
		for row in rows:
			if count == 0:
				count = count + 1
				continue
			text = row[2].value.strip().replace('\n', '').replace('\t', '').replace('\r', '').strip()
			all_file.append(text)
			count = count + 1
			
		print("All files got!!!")

		return all_file


	def getTFIDF(self):
		arr = self.allfiles()
		vectorizer=CountVectorizer(stop_words = 'english')
		transformer=TfidfTransformer()
		tfidf=transformer.fit_transform(vectorizer.fit_transform(arr))
		word=vectorizer.get_feature_names()
		weight=tfidf.toarray()
		return weight


	def calculateDistance(self, vec1, vec2):
		weight = self.getTFIDF()
		sum_Square = 0
		if vec1 == vec2:
			return 0
		for i in range(len(weight[0])):
			sum_Square += math.pow(weight[vec1][i] - weight[vec2][i], 2)
		result = math.sqrt(sum_Square)
		print(vec1, " and ", vec2, " distance is: ", result)
		return result

	def getLabels(self):
		arr = self.allfiles()
		vectorizer=CountVectorizer()
		transformer=TfidfTransformer()
		tfidf=transformer.fit_transform(vectorizer.fit_transform(arr))
		word=vectorizer.get_feature_names()
		weight=tfidf.toarray()
		Labels = []
		for w in weight:
			label = []
			index = heapq.nlargest(1, range(len(w)), w.__getitem__)
			for i in index:
				label.append(word[i].encode("utf-8"))
			Labels.append(label)
		return Labels

	# def eps_neighbor(self, a, b, eps):
	# 	dis = self.calculateDistance()
	# 	return dist[a][b] < eps

	# def region_query(self, id, eps):
	# 	seeds = []
	# 	dis = self.calculateDistance()[id]
	# 	all_distance_len = len(dis)
	# 	for i in range(all_distance_len):
	# 		if i != id:
	# 			if dis[i] < eps:
	# 				seeds.append(i)
	# 				print(i, "has been put into seed")
	# 	return seeds

	# def expand_cluster(self, clusterResult, pointId, clusterId, eps, minPts):
	# 	seeds = self.region_query(pointId, eps)
	# 	if len(seeds) < minPts: 
	# 		clusterResult[pointId] = NOISE
	# 		print(pointId, "is noise")
	# 		return False
	# 	else:
	# 		print(pointId, "is in cluster")
	# 		clusterResult[pointId] = clusterId 
	# 		for seedId in seeds:
	# 			clusterResult[seedId] = clusterId
	# 		while len(seeds) > 0: 
	# 			currentPoint = seeds[0]
	# 			queryResults = self.region_query(currentPoint, eps)
	# 			if len(queryResults) >= minPts:
	# 				for i in range(len(queryResults)):
	# 					resultPoint = queryResults[i]
	# 					if clusterResult[resultPoint] == UNCLASSIFIED:
	# 						seeds.append(resultPoint)
	# 						clusterResult[resultPoint] = clusterId
	# 					elif clusterResult[resultPoint] == NOISE:
	# 						clusterResult[resultPoint] = clusterId
	# 			seeds = seeds[1:]
	# 	return True

	def regionQuery(self, P, eps):
		neighbors = []

		all_file = self.allfiles()
		all_len = len(all_file)

		for Pn in range(0, all_len):

			if self.calculateDistance(P, Pn) < eps:
				neighbors.append(Pn)
		return neighbors



	def growCluster(self, labels, P, NeighborPts, C, eps, MinPts):
		labels[P] = C

		i = 0
		while i < len(NeighborPts):           
			Pn = NeighborPts[i]

			if labels[Pn] == -1:
				labels[Pn] = C
			elif labels[Pn] == 0:
				labels[Pn] = C

				PnNeighborPts = self.regionQuery(Pn, eps)

				if len(PnNeighborPts) >= MinPts:
					NeighborPts = NeighborPts + PnNeighborPts

			i += 1        


	def MyDBSCAN(self, eps, MinPts):
		all_file = self.allfiles()
		labels = [0]*len(all_file)

		# C is the ID of the current cluster.    
		C = 0

		for P in range(0, len(all_file)):
			print("now P is: ", P)
			if not (labels[P] == 0):
				continue

			# Find all of P's neighboring points.
			NeighborPts = self.regionQuery(P, eps)

			if len(NeighborPts) < MinPts:
				labels[P] = -1
			else: 
				C += 1
				self.growCluster(labels, P, NeighborPts, C, eps, MinPts)

		# All data has been clustered!
		return labels


	def cluster(self):
		clusters = self.MyDBSCAN(1.3, 1)
		print(clusters)
		#print("cluster Numbers = ", clusterNum)
		#self.plotFeature(clusters, clusterNum)


		
if __name__ == '__main__':
	dbs=D_clustering()
	dbs.cluster()
	#dbs.calculateDistance()
	#dbs.allfiles()


