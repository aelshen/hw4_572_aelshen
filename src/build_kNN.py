'''
#==============================================================================
build_kNN.py
/Users/aelshen/Documents/Dropbox/School/CLMS 2013-2014/Winter 2014/Ling 572-Advanced Statistical Methods for NLP/hw4_572_aelshen/src/build_kNN.py
Created on Jan 28, 2014
@author: aelshen
#==============================================================================
'''

import math
import time
import os
import sys
from collections import defaultdict
#==============================================================================
#--------------------------------Constants-------------------------------------
#==============================================================================
DEBUG = True

#==============================================================================
#-----------------------------------Main---------------------------------------
#==============================================================================
def main():
    if len(sys.argv) < 5:
        print("multinomial_NB requires five arguments:"
              + os.linesep + "\t(1)training data"
              + os.linesep + "\t(2)test data"
              + os.linesep + "\t(3)k value (1 or 2)"
              + os.linesep + "\t(4)1 for Euclid or 2 for Cosine"
              + os.linesep + "\t(5)sys_output file")
        sys.exit()
    
    if sys.argv[4] != "1" and sys.argv[4] != "2":
        print("fourth argument must be:" 
              + os.linesep + "\t1 = Euclidean"
              + os.linesep + "\t2  = Cosine")
        sys.exit()
        
    training_file = sys.argv[1]
    test_file = sys.argv[2]
    k_val = int(sys.argv[3])
    if sys.argv[4] == "1":
        SimilarityFunction = EuclideanDistance
    else:
        SimilarityFunction = CosineSimilarity
    
    sys_output = sys.argv[5]
    
    hw4 = kNN(training_file, k_val, SimilarityFunction, sys_output)
    hw4.Classify(test_file)
    
#==============================================================================    
#---------------------------------Functions------------------------------------
#==============================================================================
'''
The cosine equation is calculated as was explained by the lecture slides. 
The numerator is the sum of the product of all like features between the two 
instances. 

The denominator is the product of the magnitudes of each instance. Magnitude being
the sum of every feature^2. Since this value should be intrinsic to each training 
instance, it is precalculated to save time. 
'''
def CosineSimilarity(instance_1, instance_2):
    a = instance_1.features.copy()
    b = instance_2.features.copy()
    numerator = 0
    for feature in a:
        numerator += a[feature] * b[feature]

    cos = numerator / (instance_1.magnitude * instance_2.magnitude)
    
    return cos
'''
Euclidean distance is calculated as expected. I begin making a copy of the list of features
for each of the given instances. I got this idea from one of the gopost discussions, 
where another student who was also using defaultdict pointed out that not making a copy of the 
feature dictionary would lead to a large number of 0-value keys, which slowed performance time.

The first for loop calculates the distance betwen every feature found in a, and the feature in b.
If the feature is not in b, defaultdict automatically returns a 0.

The second for loop makes sure that any features that are in b that are not in a are factored into
the distance calculation.

I return -distance, because the larger the distance is, the less similar the two instances are. So,
since cosine similarities are positive (therefore requiring use of 'greater than', I returned 
negative distance to keep things consistent. I tried returning inverse distance (1/distance), but 
was concerned about a 0 distance comparision (i.e. identical instances). 
'''
def EuclideanDistance(instance_1, instance_2):
    a = instance_1.features.copy()
    b = instance_2.features.copy()
    distance_squared = 0
    for feature in a:
        distance_squared += math.pow(a[feature] - b[feature], 2)
    
    for feature in b:
        if feature not in a:
            distance_squared += math.pow(b[feature], 2)
    
    distance = math.sqrt(distance_squared)
    
    return -distance

def CountVotes(knn_list):
    votes = defaultdict(int)
    
    for neighbor in knn_list:
        vote = neighbor[1]
        votes[vote] += 1
        
    v=list(votes.values())
    k=list(votes.keys())
        
    return k[v.index(max(v))], votes
#==============================================================================    
#----------------------------------Classes-------------------------------------
#==============================================================================
class kNN:
    def __init__(self, training_file, k_val, SimilarityFunction, sys_output):
        self.training_instances, self.classes = self.LoadData(training_file)
        self.k_val = k_val
        self.SimilarityFunction = SimilarityFunction
        self.sys_output = open(sys_output, 'w')
        
    
    #Description    =  read in the instances form a provided data file.
    #Returns        =  instances; a list of Instance objects extracted from the 
    #                  data
    #                  classes; a dict of each class encountered paired with 
    #                  its frequency.
    def LoadData(self, file):
        instances = []
        classes = defaultdict(int)
                            
        for line in open(file, 'r').readlines():
            line = line.strip()
            if not line:
                continue
            line = line.split()
            label = line[0]
            classes[label] += 1
            features = defaultdict(int)
            for item in line[1:]:
                feat,value = item.split(":")
                features[feat] = int(value)
            
            
            instances.append( Instance(label, features) )
        
        return instances[:], classes
    
    #Description    =  classifiy a given data file
    #Returns        =  n/a
    def Classify(self, file):
        self.test_time = time.time()
        #extract all instances to be classified
        test_instances, ___ = self.LoadData(file)
        #extraneous return (useless unless for training)
        del ___
        
        self.results = []
        self.testing_tallies = defaultdict( lambda: defaultdict(int) )
        
        for instance in test_instances:
            true_label = instance.label
            knn = []
            #go through every training instance
            for neighbor in self.training_instances:
                potential_label = neighbor.label
                #calculate the distance between the current test instance
                #and the present neighbor being compared
                distance = self.SimilarityFunction(instance, neighbor)
                
                #if the knn-list is less than the specified number of neighbors k
                if len(knn) < self.k_val:
                    knn.append( (distance, potential_label) )
                #else run through every neighbor on the knn list, and if the current
                #neighbor is found to be closer, it is inserted into the proper order
                #in line
                else:
                    for i in range(len(knn)):
                        if distance > knn[i][0]:
                            knn.insert( i, (distance,potential_label) )
                            break
            
                #if the latest addition made the knn-list have more than k neighbors,
                #it is pruned.
                if len(knn) > self.k_val:
                    knn = knn[:self.k_val]
            #end neighbor in self.training_instances:
            
            #get the majority vote from the k nearest neighbors
            output_label, vote_count = CountVotes(knn)
            
            #add to list of results the true label of this test instance,
            #and the list of its k nearest neighbors
            self.results.append( (true_label, vote_count) )
            
            #used to keep track of the number of times the current test label
            #was classified as y. Used to create the confusion matrix.
            self.testing_tallies[true_label][output_label] += 1
            
        #end instance in test_instances
        self.test_time = time.time() - self.test_time
        self.PrintSys()
        self.PrintConfusionMatrix()

    def PrintSys(self):
        #from the list of classifications produced in the testing phase, 
        #run through each one, and print the true label ([0] of tuple)
        #and then run through the dict ([1] of tuple) to print the 
        #class probabilities calculated for that instance.
        for i in range( len(self.results) ):
            self.sys_output.write("test" + str(i) + ": " + self.results[i][0])
            for cls in self.classes:
                self.sys_output.write("\t" + cls + "\t" + str(self.results[i][1][cls] / self.k_val))    
            self.sys_output.write(os.linesep)          


    def PrintConfusionMatrix(self):
            print("Confusion matrix for the test data:")
            print("row is the truth, column is the system output" + os.linesep)

            correct_labels = 0
            total_labels = 0

            print("\t"*2, end="")
            for cls in self.classes:
                    print(cls + "\t", end="")
            print("")
            for true_class in self.classes:
                    print(true_class + "\t", end="")
                    #from the training_testing tallies produced in Test()
                    #print the number of times the true label was classified as Y
                    #for each y in self.classes
                    for knn_class in self.classes:
                            print(str(self.testing_tallies[true_class][knn_class]) + "\t", end="")
                            total_labels += self.testing_tallies[true_class][knn_class]
                            if true_class == knn_class:
                                    correct_labels += self.testing_tallies[true_class][knn_class]
                    print("")

            print(os.linesep + " Test accuracy="+str(float(correct_labels)/total_labels) + os.linesep)
            print("Test time: " + str(self.test_time))

class Instance:
    def __init__(self, label, features):
        self.label = label
        self.features = features
        self.magnitude = self.CalcMagnitude()
    
    #Description    =  calculate the magnitude, only used in Cosine Similarity
    #Returns        =  sum; the magnitude used in calculating Cosine Similarity
    def CalcMagnitude(self):
        sum = 0
        for feat in self.features:
            sum += math.pow(self.features[feat], 2)
        
        return math.sqrt(sum)


if __name__ == "__main__":
    sys.exit( main() )