import os
import string
import math
import numpy as np
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


class Cluster:

    def __init__(self):
        # print("init")
        self.x_part = []
        self.y_part = []
        self.predictions_dict={}
        self.mapping={}
        self.tfvec = TfidfVectorizer(stop_words='english')
        self.k_clusters = 5



    def cluster(self, directory_path):

        index=0
        for _, _, files in os.walk(directory_path):
            for file in files:
                file = "/"+file
                # print(file)
                with open(directory_path+file, 'r', encoding="utf8", errors='ignore') as f:
                    temp = ""
                    line = f.read()
                    line = line.replace('\n', ' ')
                    line = line.strip('\t')
                    temp += line
                temp = temp.translate(str.maketrans('', '', string.punctuation))
                temp = temp.lower()
                # print(temp)
                self.x_part.append(temp)
                label = -1
                # label = file.split('_')[1]
                # label = label[0]
                self.y_part.append(label)
                file = file.replace("/","")
                self.mapping[index] = file
                self.predictions_dict[file] = []
                index += 1

        train_vec = self.tfvec.fit_transform(self.x_part)
        train_vec = train_vec.toarray()
        dim = train_vec.shape[1]
        

        tolerance = 0.001
        g_dict = {}
        trials = 1
        best_list = []
        best_acc = -100
        samples = train_vec.shape[0]

        for trial in range(trials):
            k_dict={}
            k_means = np.random.uniform(size=(self.k_clusters, dim))
            for row in range(self.k_clusters):
                k_means[row,:] /= np.linalg.norm(k_means[row,:])
            while True:
                my_mean = [-1]*samples
                points_in_mean = {new_list: [] for new_list in range(self.k_clusters)} 
                for point in range(samples):
                    min_dist = math.inf
                    assgn_centroid = -1
                    for mean in range(k_means.shape[0]):
                        dist = np.linalg.norm(train_vec[point,:] - k_means[mean,:])
                        if dist < min_dist:
                            min_dist = dist
                            assgn_centroid = mean
                    my_mean[point] = assgn_centroid
                    points_in_mean[assgn_centroid].append(point)
                g_dict = points_in_mean
                new_means = np.zeros(k_means.shape)
                for point in range(train_vec.shape[0]):
                    assgn_centroid = my_mean[point]
                    new_means[assgn_centroid] += train_vec[point]
                
                for mean in range(new_means.shape[0]):
                    new_means[mean] /= len(points_in_mean[mean])
                distances = np.linalg.norm(k_means-new_means, axis=1)
                dist = sum(distances)
                k_means = new_means
                if dist < tolerance:
                    break
            predictions=[-1]*train_vec.shape[0]
            for mean in range(self.k_clusters):
                actual_labels = [self.y_part[g_dict[mean][ind]] for ind in range(len(g_dict[mean]))]
                # label = statistics.mode(actual_labels)
                label=mean
                for point in g_dict[mean]:
                    predictions[point] = label
            acc = accuracy_score(self.y_part, predictions)*100
#             print("trial accuracy "+str(trial)+" "+str(acc))
            if acc > best_acc:
                best_acc = acc
                best_list = predictions
        

        for p in range(len(best_list)):
            self.predictions_dict[self.mapping[p]] = best_list[p]

        # print(self.predictions_dict)
        return self.predictions_dict



