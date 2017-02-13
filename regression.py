import numpy as np

def calc(degree,xx,yy):

	x = np.asarray(xx)
	y = np.transpose(np.matrix(yy))
	X = np.matrix([[i**j for j in range(degree+1)] for i in x])

	if len(x) <= degree:
		return "ERROR: Degree must be less than the number of data points given!"

	result = list(np.asarray(np.transpose(np.linalg.inv(np.transpose(X)*X)*(np.transpose(X)*y)))[0])

	return [round(result[i],3) for i in range(len(result))]

def calc_exact(degree,xx,yy):

	x = np.asarray(xx)
	y = np.transpose(np.matrix(yy))
	X = np.matrix([[i**j for j in range(degree+1)] for i in x])

	if len(x) <= degree:
		return "ERROR: Degree must be less than the number of data points given!"

	return np.linalg.inv(np.transpose(X)*X)*(np.transpose(X)*y)

def calc_int(degree,xx,yy):

	x = np.asarray(xx)
	y = np.transpose(np.matrix(yy))
	X = np.matrix([[i**j for j in range(degree+1)] for i in x])

	if len(x) <= degree:
		return "ERROR: Degree must be less than the number of data points given!"

	print np.linalg.inv(np.transpose(X)*X)
	print (np.transpose(X)*y)
	return np.linalg.inv(np.transpose(X)*X)*(np.transpose(X)*y)


def calc_2var(xx,yy,zz):

	x = np.asarray(xx)
	y = np.asarray(yy)
	z = np.transpose(np.matrix(zz))
	X = np.matrix([[1,x[i]*y[i],x[i] * y[i]**2,x[i]**2 * y[i],x[i]**2 * y[i]**2] for i in range(len(x))])

	result = list(np.asarray(np.transpose(np.linalg.inv(np.transpose(X)*X)*(np.transpose(X)*z)))[0])

	return [round(result[i],3) for i in range(len(result))]

def lazy():

	#up, down, left, right, up-right, up-left, down-right, down-left

	y = np.transpose(np.matrix([1,-1,1,-1,0,1,0,-1]))
	X = np.matrix([
[1,0,0,0],
[0,1,0,0],
[0,0,1,0],
[0,0,0,1],
[1,0,0,1],
[1,0,1,0],
[0,1,0,1],
[0,1,1,0],
])

	result = list(np.asarray(np.transpose(np.linalg.inv(np.transpose(X)*X)*(np.transpose(X)*y)))[0])

	return result#[round(result[i],3) for i in range(len(result))]

def surd(x,n):
        if x < 0:
                return -(-x)**(1./n)
        else:
                return x**(1./n)

def classification(xx,yy):
        degree = 1
        x = np.asarray(xx)
	y = np.transpose(np.matrix(yy))
	X = np.matrix([[1,surd(i,3),surd(i,5),surd(i,7),surd(i,9),surd(i,11)] for i in x])

	if len(x) <= degree:
		return "ERROR: Degree must be less than the number of data points given!"

	result = list(np.asarray(np.transpose(np.linalg.inv(np.transpose(X)*X)*(np.transpose(X)*y)))[0])

	return [round(result[i],3) for i in range(len(result))]

print calc_exact(3,[1,2,3,4,5],[0,0,0,1,1])
