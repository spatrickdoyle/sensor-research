import time,math,glob
import numpy as np
#import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy.special as sp
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
#import regression as r
#import analysis


PI = 3.1415926535
E = 2.71828
step = lambda x: 0.0 if x < 0 else 1.0 if x > 0 else 0.5

def c_(n,m,T,w,X,Y):
    return -(1/(1j*n*w*T))*sum([((Y[j]-Y[j-1])/(X[j]-X[j-1]))*(X[j]*(E**(-1j*n*w*X[j])) - X[j-1]*(E**(-1j*n*w*X[j-1]))) + (E**(-1j*n*w*X[j]) - E**(-1j*n*w*X[j-1]))*(Y[j]-((Y[j]-Y[j-1])/(X[j]-X[j-1]))*X[j]+(1/(1j*n*w))*((Y[j]-Y[j-1])/(X[j]-X[j-1]))) for j in range(1,m)])

def f_(x,mu,sig):
    return 1 - 0.5*( sp.erf(np.abs(x-mu)/(sig*math.sqrt(2))) - sp.erf(-np.abs(x-mu)/(sig*math.sqrt(2))) )

class Classification:
    def __init__(self,classification):
        if not isinstance(classification,list):
            self.classification = [[classification for j in range(8)] for i in range(8)]
        else:
            self.classification = classification

    def set(self,i,j,clas):
        self.classification[i][j] = clas

    def get(self,i,j):
        return self.classification[i][j]

class Sweep:
    def __init__(self,capfile,cndfile,classification=None):
        capFile = file(capfile)
        cndFile = file(cndfile)

        #Construct the sweep matrices - an 8x8 matrix, where each element is a 401 length array (a sweep)
        data = []
        for line in capFile.readlines():
            data.append([float(i) for i in line.split(',')])
        self.capacitance = [[[] for j in range(8)] for i in range(8)]
        for sweep in range(len(data)):
            self.capacitance[sweep%8][sweep/8] = data[sweep]

        data = []
        for line in cndFile.readlines():
            data.append([float(i) for i in line.split(',')])
        self.conductivity = [[[] for j in range(8)] for i in range(8)]
        for sweep in range(len(data)):
            self.conductivity[sweep%8][sweep/8] = data[sweep]

        #For now just use the first intersection
        #self.capacitance = [[[float(i) for i in capFile.readline().split(',')]]]
        #self.conductivity = [[[float(i) for i in cndFile.readline().split(',')]]]

        capFile.close()
        cndFile.close()

        #Generate complex data set
        self.data = [[[self.capacitance[i][j][k]+(self.conductivity[i][j][k]*1j) for k in range(len(self.capacitance[i][j]))] for j in range(len(self.capacitance[i]))] for i in range(len(self.capacitance))]

        #The classification, if there is one
        self.classification = classification

    def setClass(self,c):
        #set the classification for this sweep
        self.classification = c

    def c(self,n):
        #return order n coefficients for this sweep as a matrix of complex numbers
        m = 401
        X = range(401)
        C = []
        for i in self.data:
            tmp = []
            for Y in i:
                T = X[m-1]-X[0]
                w = (2*PI)/T
                tmp.append(c_(n,m,T,w,X,Y))
            C.append(tmp)
        return C

    def g(self,x,order):
        m = 401
        X = range(401)
        T = X[m-1]-X[0]
        w = (2*PI)/T
        return sum([ (self.c(n)[0][0])*(E**(1j*n*w*x)) for n in range(-order,0)]) + sum([ (self.c(n)[0][0])*(E**(1j*n*w*x)) for n in range(1,order+1)])

class Model:
    #Each model is loaded with sweeps from the same classification and generates probabilities that sweeps belong to that given classification
    def __init__(self,order,classification):
        #The model itself is an 8x8 matrix, and each index is a list of complex mu-sigma tuples
        self.order = order
        self.model = [[[(0,0) for i in range(order)] for j in range(8)] for k in range(8)]
        self.classification = classification

    def train(self,sweeps):
        #Train each point in the models using an array of sweep objects

        #generate the model for each sweep
        for k in range(8):
            for l in range(8):
                print k,l
                for j in range(self.order):
                    coefs = [s.c(j+1)[k][l] for s in sweeps]
                    mean = sum(coefs)/len(coefs)
                    sigma = np.sqrt(sum([np.real(i)**2 for i in coefs])/len(coefs) - mean**2)
                    self.model[k][l][j] = (mean,sigma)
        #print self.model

    def test(self,sweeps):
        #Return a list of probabilities, one for each sweep
        for k in range(8):
            for l in range(8):
                for s in sweeps:
                    print "Sweep:"
                    coefs = [s.c(i+1)[k][l] for i in range(self.order)] #List of coefficients for the test sweep, where the order is the index+1
                    means = [i[0] for i in self.model[k][l]]
                    sigs = [i[1] for i in self.model[k][l]]

                    #for order in range(self.order):
                    #print f_(-1.06322e-3,-1.03934e-3,2.647e-5)
                    #print coefs[order],means[order],sigs[order]
                    #print f_(np.real(coefs[order]),np.real(means[order]),sigs[order])
                    print k,l,sum([np.real(f_(np.real(coefs[order]),np.real(means[order]),sigs[order])) for order in range(self.order)])/self.order
            #print ''

    def g(self,x,order):
        m = 401
        X = range(401)
        T = X[m-1]-X[0]
        w = (2*PI)/T
        return sum([ (-self.model[0][0][-(n+1)][0])*(E**(1j*n*w*x)) for n in range(-order,0)]) + sum([ (self.model[0][0][n-1][0])*(E**(1j*n*w*x)) for n in range(1,order+1)])
