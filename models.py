import math
import numpy as np
import scipy.special as sp

PI = 3.1415926535
E = 2.71828

#Heaviside step function, half-maximum convention
step = lambda x: 0.0 if x < 0 else 1.0 if x > 0 else 0.5

#General formula for calculating the nth order Fourier coefficient
def c_(n,m,T,w,X,Y):
    return -(1/(1j*n*w*T))*sum([((Y[j]-Y[j-1])/(X[j]-X[j-1]))*(X[j]*(E**(-1j*n*w*X[j])) - X[j-1]*(E**(-1j*n*w*X[j-1]))) + (E**(-1j*n*w*X[j]) - E**(-1j*n*w*X[j-1]))*(Y[j]-((Y[j]-Y[j-1])/(X[j]-X[j-1]))*X[j]+(1/(1j*n*w))*((Y[j]-Y[j-1])/(X[j]-X[j-1]))) for j in range(1,m)])

def c0_(m,T,X,Y):
    return (1/(2*T))*sum([(Y[j]-Y[j-1])*(X[j-1]-X[j])+Y[j] for j in range(1,m)])

#General formula for finding the probability a value falls within a normal distribution
def f_(x,mu,sig):
    return (E**(-(np.absolute(x)**2)/(2*(sig**2))))/(2*PI*(sig**2))
        

class Sweep:
    def __init__(self,fileName,size):
        self.size = size #The size of the matrix to load and store

        #Open the two files, which should have the exact same path except with 'cap' or 'con' in the placeholder
        capFile = file(fileName%'cap') #Capacitance data
        conFile = file(fileName%'con') #Conductivity data

        #Construct the sweep matrices - 8x8 matrix, where each element is a 401 length array (a sweep)
        data = []
        for line in capFile.readlines():
            data.append([float(i) for i in line.split(',')])
        size = int(math.sqrt(len(data)))
        capData = [[[] for j in range(size)] for i in range(size)]
        for sweep in range(len(data)):
            capData[sweep%size][sweep/size] = data[sweep]

        data = []
        for line in conFile.readlines():
            data.append([float(i) for i in line.split(',')])
        size = int(math.sqrt(len(data)))
        conData = [[[] for j in range(size)] for i in range(size)]
        for sweep in range(len(data)):
            conData[sweep%size][sweep/size] = data[sweep]

        capFile.close()
        conFile.close()

        #Generate complex data subset using all the data just loaded
        self.data = [[[capData[i][j][k]+(conData[i][j][k]*1j) for k in range(len(capData[i][j]))] for j in range(self.size)] for i in range(self.size)]


        #Generate a matrix of classifications (a separate list for each intersection) as well as the likelihoods of each classification

        self.classifications = [[[] for i in range(size)] for j in range(size)] #Matrix of possible classifications (lists of strings)
        self.likelihoods = [[[] for i in range(size)] for j in range(size)] #Matrix of the likelihoods for each classification in the classifications matrix (list of floats 0.0-1.0)

    def addClass(self,i,j,clas,likelihood):
        #Add a possible classification and its likelihood to a single index of the matrix
        self.classifications[i][j].append(clas)
        self.likelihoods[i][j].append(likelihood)

    def getClass(self,i,j):
        return self.classifications[i][j][self.likelihoods[i][j].index(max(self.likelihoods[i][j]))]

    def genClass(self):
        return [[self.getClass(i,j) for j in range(self.size)] for i in range(self.size)]

    def c(self,n,row,col):
        #return order n coefficients for this sweep as a matrix of complex numbers
        m = 401
        X = range(401)
        C = []
        T = X[m-1]-X[0]
        w = (2*PI)/T

        Y = self.data[row][col]
        if n != 0:
            C = c_(n,m,T,w,X,Y)
        else:
            C = c0_(m,T,X,Y)
        return C


class Model:
    #Each model is loaded with sweeps from the same classification and generates probabilities that sweeps belong to that given classification
    def __init__(self,order,clas):
        #The model itself is a max 8x8 matrix, and each index is a list of complex mu-sigma tuples
        self.model = [[[(0,0) for i in range(order)] for j in range(8)] for k in range(8)]

        self.order = order #The order of the model (the maximum n value for generated coefficients)
        self.classification = clas #String representation of the classification of this model

        self.rang = (1,self.order+1) #Exclusive range for orders of coefficients to use (so that we can easily include or exclude 0)

    def train(self,sweeps):
        #Train each point in the models using an array of sweep objects

        print "Training %s model..."%self.classification

        #generate the model for each sweep
        for row in range(sweeps[0].size):
            for col in range(sweeps[0].size):
                for order in range(self.rang[0],self.rang[1]):
                    #Generate a list of coefficients of a specific order for each sweep
                    coefs = [s.c(order,row,col) for s in sweeps]
                    #Find the mean of this nth order coefficient
                    mean = np.average(coefs)#sum(coefs)/len(coefs)
                    #And the standard deviation
                    sigma = math.sqrt(0.5*(sum([np.real(i)**2 for i in coefs])/len(coefs) + sum([np.real(i)**2 for i in coefs])/len(coefs)))

                    #Save it into a big matrix
                    self.model[row][col][order-1] = (mean,sigma)

    def test(self,sweeps):
        #Return a list of probabilities, one for each sweep
        for s in sweeps:
            for row in range(sweeps[0].size):
                for col in range(sweeps[0].size):
                    coefs = [s.c(i,row,col) for i in range(self.rang[0],self.rang[1])] #List of coefficients for the test sweep, where the order is the index+1
                    means = [i[0] for i in self.model[row][col]]
                    sigs = [i[1] for i in self.model[row][col]]

                    likelihood = sum([f_(coefs[order],means[order],sigs[order]) for order in range(self.rang[0]-1,self.rang[1]-1)])/self.order

                    s.addClass(row,col,self.classification,likelihood)
