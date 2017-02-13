import pygame,time,math,glob
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import regression as r


class Sweep:
    def __init__(self,capfile,cndfile):
        capFile = file(capfile)
        cndFile = file(cndfile)

        #Construct the sweep matrices
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

        capFile.close()
        cndFile.close()

    def getCapacitance(self):
        #return whole sweep matrix
        return self.capacitance

    def getCapacitance(self,col,row):
        #return a list, the whole sweep at the given point
        return self.capacitance[row][col]

    def getConductivity(self):
        #return whole sweep matrix
        return self.conductivity

    def getConductivity(self,col,row):
        #return a list, the whole sweep at the given point
        return self.conductivity[row][col]

class Model:
    #Model(class,[list,of,args])
    def __init__(self,model,args,name):
        #Create two matrices of models, one for capacitance data and one for conductivity data
        if len(args) == 0:
            self.capacitance = [[model() for i in range(8)] for j in range(8)]
            self.conductivity = [[model() for i in range(8)] for j in range(8)]
        elif len(args) == 1:
            self.capacitance = [[model(args[0]) for i in range(8)] for j in range(8)]
            self.conductivity = [[model(args[0]) for i in range(8)] for j in range(8)]

        self.name = name;

    def train(self,sweeps,classifications):
        #Train each point in the models using an array of sweeps and an array of associated classifications
        '''for i in range(8):
            for j in range(8):
                self.capacitance[j][i].fit([sweeps[k].getCapacitance(i,j) for k in range(len(sweeps))],classifications)'''

        for i in range(8):
            for j in range(8):
                self.conductivity[j][i].fit([sweeps[k].getConductivity(i,j) for k in range(len(sweeps))],classifications)

    def testCapacitance(self,sweeps):
        #return a list of classifications the model thinks each sweep falls into based on capacitance
        ret = [[[] for j in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                ret[j][i] = self.capacitance[j][i].predict([sweeps[k].getCapacitance(j,i) for k in range(len(sweeps))]).tolist()
        return ret

    def testConductivity(self,sweeps):
        #return a list of classifications the model thinks each sweep falls into based on conductivity
        ret = [[[] for j in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                ret[j][i] = self.conductivity[j][i].predict([sweeps[k].getConductivity(j,i) for k in range(len(sweeps))]).tolist()
        return ret


class Regressed:
    #def __init__(self,base):
    #    self.base = base
    def __init__(self):
        self.classes = []
        self.sent = 0

    def fit(self,X,y):
        for x in range(len(X)):
            if self.sent == 1:
                #For conductivity data
                regressed = [r.calc_exact(3,range(len(X[x])),np.negative(np.log(np.abs(X[x])))),y[x]]
                self.sent = 0
            else:
                #For capacitance data
                regressed = [r.calc_exact(3,range(len(X[x])),X[x]),y[x]]
                self.sent = 1
            #regressed = [r.calc_exact(3,range(len(X[x])),np.negative(np.log(np.abs(X[x])))),y[x]]
            #print regressed
            self.classes.append(regressed)#Capacitance: cubic, most important term is intercept, Conductivity: take the abs and then the log and then the negative of the data, and fit that to a cubic
            '''ys = [sum([regressed[0].tolist()[i][0]*(j**i) for i in range(len(regressed[0].tolist()))]) for j in range(len(X[x]))]
            plt.plot(range(len(X[x])),np.negative(np.log(np.abs(X[x]))),'b')
            plt.plot(range(len(X[x])),ys,'r')
            plt.show()
            raw_input()'''

    def predict(self,tests):
        results = []
        for t in tests:
            test = r.calc_exact(3,range(len(t)),t)
            minn = [self.classes[0][1],np.linalg.norm(self.classes[0][0]-test)]
            for clas in self.classes:
                diff = np.linalg.norm(clas[0]-test)
                if diff < minn[1]:
                    minn = [clas[1],diff]
            #print minn[0]
            results.append(minn[0])
            #raw_input()
        return np.asarray(results)


size = 300.0


#Open data files - 1 air, 2 apple, 3 Dr. Sabuncu's hand, 4 bottom half apple, 5 Sean's hand
#6 February - 1 air, 2 air, 3 air, 4 copper, 5 Sean's hand, 6 Sean's finger on the top half, 7 Shahriar's hand
print "Opening data..."
CapFileNames = sorted(glob.glob("6Feb/A*.csv"))
CndFileNames = sorted(glob.glob("6Feb/B*.csv"))

trainingData = [Sweep(CapFileNames[i],CndFileNames[i]) for i in range(5)]
testingData = [Sweep(CapFileNames[i],CndFileNames[i]) for i in range(5,7)]

#Create sets of models and train them
print "Training models..."
#Air: 0.25, Copper: 0.5, Sean's hand: 0.75
y = [0.25,0.25,0.25,0.5,0.75]
models = [Model(Regressed,[],"My regression model")]#,Model(svm.SVC,[],"Support vector machine"),Model(RandomForestClassifier,[10],"Random forest - 10 nodes")]#,Model(RandomForestClassifier,[100])]
for m in models:
    m.train(trainingData,y)

#Test remaining data
print "Testing..."
fig, axes = plt.subplots(figsize=[8.0, 7.5], ncols=3, nrows=2)
for m in range(len(models)):
    print models[m].name
    print "Capacitance:"
    data = np.asarray(models[m].testCapacitance(testingData))
    axes[0,m].imshow(data[:,:,0],interpolation='none')
    print "Conductivity:"
    data = np.asarray( models[m].testConductivity(testingData))
    axes[1,m].imshow(data[:,:,0],interpolation='none')

#axes[0, 1].imshow(small_im, interpolation='nearest')
#axes[1, 0].imshow(small_im, interpolation='none')
#axes[1, 2].imshow(small_im, interpolation='nearest')
fig.subplots_adjust(left=0.24, wspace=0.2, hspace=0.1,
                    bottom=0.05, top=0.86)

# Label the rows and columns of the table
fig.text(0.03, 0.645, 'Capacitance', ha='left')
fig.text(0.03, 0.225, 'Conductivity', ha='left')
fig.text(0.383, 0.90, "Interpolation = 'none'", ha='center')
fig.text(0.75, 0.90, "Interpolation = 'nearest'", ha='center')

red_patch = mpatches.Patch(color=(1,0,0), label='Air')
plt.legend(handles=[red_patch])

plt.show()


'''#Create window
pygame.init()
screen = pygame.display.set_mode((int(size*3),int(size*2)))

while True:
    screen.fill((0,0,0))
    for trial in range(2):
        print "Test %d..."%trial
        for model in range(len(outputCap)):
            for row in range(len(outputCap[model])):
                for col in range(len(outputCap[model][row])):
                    if outputCap[model][row][col][trial] == "air":
                        color = (0,0,255)
                    else:
                        color = (255,0,0)
                    pygame.draw.rect(screen,color,(model*size + col*(size/8.0),row*(size/8.0),size/8.0,size/8.0))

                    if outputCnd[model][row][col][trial] == "air":
                        color = (0,0,255)
                    else:
                        color = (255,0,0)
                    pygame.draw.rect(screen,color,(model*size + col*(size/8.0),size + row*(size/8.0),size/8.0,size/8.0))

        for i in range(12):
            pygame.draw.rect(screen,(0,0,0),((2*i + 1)*(size/8.0),0,1,screen.get_height()))
        for i in range(8):
            pygame.draw.rect(screen,(0,0,0),(0,(2*i + 1)*(size/8.0),screen.get_width(),2))

        for i in range(2):
            pygame.draw.rect(screen,(0,255,0),((i+1)*size,0,1,screen.get_height()))
        for i in range(1):
            pygame.draw.rect(screen,(0,255,0),(0,(i+1)*size,screen.get_width(),2))

        pygame.display.update()
        raw_input()'''
