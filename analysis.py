from __future__ import division
from math import cos,sin
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import random

def b_simple(n,x,y):
    a = 3.1415926*n #Pi and n only ever appear multiplied by each other
    m = len(x) #Number of data points

    lead = 2/(a**2) #Leading coefficient
    term1 = (y[0]/(x[0]+0.5)) * ( -a*(x[0]+0.5)*cos(a*(x[0]+0.5)/(x[m-1]+1)) + (x[m-1]+1)*sin(a*(x[0]+0.5)/(x[m-1]+1)) )
    term2 = y[m-1]*( a*cos(a*(x[m-1]+0.5)/(x[m-1]+1)) - 2*(x[m-1]+1)*(sin(a)-sin(a*(x[m-1]+0.5)/(x[m-1]+1))) )
    term3 = sum([ (x[m-1]+1)*((y[i]-y[i-1])/(x[i]-x[i-1]))*(sin(a*(x[i]+0.5)/(x[m-1]+1))-sin(a*(x[i-1]+0.5)/(x[m-1]+1))) + a*(y[i-1]-y[i])*cos(a*(x[i-1]+0.5)/(x[m-1]+1)) for i in range(1,m)])

    return lead*(term1+term2+term3)


def b_aliased(n,x,y):
    m = len(x) #Number of data points
    a = 3.1415926*n #Pi and n only ever appear multiplied by each other
    b = x[0]+0.5
    c = y[0]/b
    d = lambda i: x[i]+0.5
    e = lambda i: x[i-1]+0.5
    f = lambda i: (y[i]-y[i-1])/(x[i]-x[i-1])
    g = lambda i: y[i]
    h = y[m-1]
    j = x[m-1]+1

    result = (1/(0.5+j))*( (1/(a**2))*c*(a*b*(-1-(2*j))*cos((a*b)/(0.5+j))+2*((0.5+j)**2)*sin((a*b)/(0.5+j))) - (1/(a**2))*2*h*(a*(-0.25-(0.5*j))*cos((a*j)/(0.5+j))+((0.5+j)**2)*(sin(a)-sin((a*j)/(0.5+j)))) - (1/(a**2))*2*h*(a*(-0.25-(0.5*j))*cos((a*j)/(0.5+j))+((0.5+j)**2)*(sin(a)-sin((a*j)/(0.5+j)))) + sum([ (1/(a**2))*(0.5+j)*( a*cos((a*d(i))/(j+0.5))*g(i) - a*cos((a+e(i))/(0.5+j))*((d(i)-e(i))*f(i)+g(i)) + (0.5+j)*f(i)*sin((a*d(i))/(0.5+j)) - (0.5+j)*f(i)*sin((a*e(i))/(0.5+j)) ) for i in range(1,m-1)]) + sum([ (1/(a**2))*(0.5+j)*( -a*cos((a*d(i))/(j+0.5))*g(i) + a*cos((a+e(i))/(0.5+j))*((-d(i)+e(i))*f(i)+g(i)) + (0.5+j)*f(i)*sin((a*d(i))/(0.5+j)) - (0.5+j)*f(i)*sin((a*e(i))/(0.5+j)) ) for i in range(1,m-1)]))

    return result


def step(x):
    if x < 0:
        return 0
    elif x > 0:
        return 1
    else:
        return 0.5

def b_integrated(nn,X,Y):
    PI = 3.14159265
    m = len(X)

    T = 2*(X[m-1]+1)
    w = (2*PI)/T

    #term1 = lambda n: (Y[0]/(X[0]+0.5))*(integrate.quad(lambda x: x*sin(n*w*x),0,X[0]+0.5)[0])
    term1 = lambda n: ((Y[0]*(X[m-1]+1))/(PI*n*(X[0]+0.5)))*(((X[m-1]+1)/(n*PI))*sin(n*PI*(X[0]+0.5)/(X[m-1]+1)) - (X[0]+0.5)*cos(n*PI*(X[0]+0.5)/(X[m-1]+1)))
    term2 = lambda n: sum([ integrate.quad(lambda x: (((Y[i]-Y[i-1])/(X[i]-X[i-1]))*(x-(X[i]+0.5))+Y[i])*sin(n*w*x),X[i-1]+0.5,X[i]+0.5)[0] for i in range(1,m)])
    term3 = lambda n: -(2*Y[m-1])*integrate.quad(lambda x: (x-(X[m-1]+1))*sin(n*w*x),X[m-1]+0.5,X[m-1]+1)[0]

    return (4/T)*(term1(nn)+term2(nn)+term3(nn))


def graphModel(order,X,Y):
    PI = 3.14159265

    for i in range(len(X)):
        m = len(X[i])
        T = 2*(X[i][m-1]+1)
        w = (2*PI)/T

        d = 1000
        F = lambda x: sum([b_integrated(n,X[i],Y[i])*sin(n*w*x) for n in range(1,order+1)])
        plt.plot([j/d for j in range(d*int(X[i][-1]+2))],[F(x) for x in [j/d for j in range(d*int(X[i][-1]+2))]])

def graphDist(order,X,Y):
    '''X = np.asarray([1, 2, 3, 4, 4.4])
    Y = np.asarray([34, -4.3, 6, -2.1, 32])
    print X
    print Y
    print b_integrated(1,X,Y)
    print ''
    '''
    b1s = []
    #b2s = []

    for i in range(len(X)):
        #X_ = random.randint(5,15)*0.1*(X+np.asarray([random.randint(-39,39)*0.01 for j in range(len(X))]))
        #Y_ = random.randint(5,15)*0.1*(Y+np.asarray([random.randint(-39,39)*0.01 for j in range(len(Y))]))

        #print X_
        #print Y_
        b1s.append(b_integrated(order,X[i],Y[i]))
        #b2s.append(b_integrated(2,X_,Y_))
        print b1s[-1]
        #print b2s[-1]
        #print ''

    print "Mean: %f"%(sum(b1s)/len(b1s))
    plt.plot(b1s,[0]*len(b1s),'ro')
    #plt.plot(range(4,25),[1/(max(b1s)-min(b1s)) if (i <= max(b1s) and i >= min(b1s)) else 0 for i in range(4,25)],'g-')
    #plt.plot(b2s,[1]*len(b2s),'bo')

#s = lambda x: (Y[0]/(X[0]+0.5))*x*step(-x*(x-(X[0]+0.5))) + sum([ (((Y[n]-Y[n-1])/(X[n]-X[n-1]))*(x-(X[n]+0.5))+Y[n])*step(-(x-(X[n]+0.5))*(x-(X[n-1]+0.5))) for n in range(1,m)]) - 2*Y[m-1]*(x-(X[m-1]+1))*step(-(x-(X[m-1]+0.5))*(x-(X[m-1]+1)))

#f = lambda x: s(x)-s(-x)

#size = 0.1
#plt.plot([size*i for i in range(-int((X[m-1]+1)/size),1+int((X[m-1]+1)/size))],[f(size*i) for i in range(-int((X[m-1]+1)/size),1+int((X[m-1]+1)/size))])
#plt.show()

if __name__ == "__main__":
    file1 = file('sound_data/seanee1.dat','r')
    data1 = file1.readlines()[2:]
    #file2 = file('sound_data/seanee2.dat','r')
    #file3 = file('sound_data/seanee3.dat','r')

    X = [float(i.split()[0]) for i in data1]
    Y = [float(i.split()[1]) for i in data1]
    #print X
    #print Y

    '''print b_integrated(1,X,Y)
    print b_integrated(2,X,Y)
    print b_integrated(3,X,Y)'''

    plt.plot([i+0.5 for i in X],Y,'r-')
    graphModel(5,[X],[Y])
    plt.show()
