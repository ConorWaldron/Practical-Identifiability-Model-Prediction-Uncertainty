# -*- coding: utf-8 -*-
#import pyDOE
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
import time
#from scipy.optimize import minimize
#import openpyxl
#import pandas as pd
#import scipy.stats as stats
#import os
#import openpyxl

#from mpl_toolkits.mplot3d import Axes3D

start=time.time()

CatalystMass= 0.1156
CatalystDiameter = 950
SingleExpCondition = [100, 20, 1.0]

SampleNumber=10000

FactorialParameterValues=[16.65, 6.40, 0.28]
MDParameterValues=[16.60, 6.33, 0.48]
MBDoEParameterValues=[16.66, 6.88, 0.53]

##########################################################################################################################################
#If you assume the paraemter values are correlated  ######################################################################################

FactorialCovariance=[[0.0043679, -0.00790276, -0.01018036],[-0.00790276,0.17174359, 0.05094372],[-0.01018036,0.05094372, 0.03245691]]
MDCovariance=[[0.00261499,0.00192817,-0.00440308],[0.00192817,0.12547173,0.02296409],[-0.00440308,0.02296409,0.01400866]]
MBDoECovariance=[[0.00164276,-0.00258033,-0.00342592],[-0.00258033,0.04907792,0.01422575],[-0.00342592,0.01422575,0.00945524]]

FactorialGeneratedParameters=np.random.multivariate_normal(FactorialParameterValues, FactorialCovariance, SampleNumber)
DiscriminationGeneratedParameters=np.random.multivariate_normal(MDParameterValues, MDCovariance, SampleNumber)
MBDoEGeneratedParameters=np.random.multivariate_normal(MBDoEParameterValues, MBDoECovariance, SampleNumber)

#Plot Histograms
#plt.figure(1)
#bins= np.linspace(-0.5,18,1000)
#plt.hist(FactorialGeneratedParameters[:,0], bins, normed=True, color='r', alpha=0.5, label='Parameter estimates after factorial') #alpha is transparency
#plt.hist(FactorialGeneratedParameters[:,1], bins, normed=True, color='r', alpha=0.5) #alpha is transparency
#plt.hist(FactorialGeneratedParameters[:,2], bins, normed=True, color='r', alpha=0.5)
#plt.hist(DiscriminationGeneratedParameters[:,0], bins, normed=True, color='b', alpha=0.5, label='Parameter estimates after model discrimination')
#plt.hist(DiscriminationGeneratedParameters[:,1], bins, normed=True, color='b', alpha=0.5)
#plt.hist(DiscriminationGeneratedParameters[:,2], bins, normed=True, color='b', alpha=0.5)
#plt.hist(MBDoEGeneratedParameters[:,0], bins, normed=True, color='g', alpha=0.5, label='Parameter estimates after MBDoE')
#plt.hist(MBDoEGeneratedParameters[:,1], bins, normed=True, color='g', alpha=0.5)
#plt.hist(MBDoEGeneratedParameters[:,2], bins, normed=True, color='g', alpha=0.5)
#plt.ylim(0,8.5)
#plt.xlabel('Parameter Value')
#plt.ylabel('Observed Frequency')
#plt.legend(loc='best')
#plt.show()

#Generate data for each parameter value
def kinetic_model(c,W,u,d,theta):
    CBA = c[0] #mol/L
    CEtOH = c[1]
    CEB = c[2] #mol/L
    CW = c[3]
    
    tempC = u[0] #oC
    flowuLmin = u[1] #uL/min
    InletC = u[2] #mol/L
    dSphere = d*10**(-6) #um
        
    KP1 = theta[0]
    KP2 = theta[1]
    KW  = theta[2]
    #KEtOH = theta[3]
    R = 8.314
    #pi = math.pi
    #Tort = 3.745
    
    flowLs = flowuLmin*10**(-6)/60 #L/s
    TK=tempC+273.15 #K
    TM=((140+70)/2)+273.15 #K
    #rSphere = dSphere/2 #m
    
    #TubeDiameter = 1000*10**(-6) #m
    #TubeArea = pi*(TubeDiameter/2)**(2) #m2
    #SuperVelocity = (flowLs/1000)/TubeArea #m/s
    
    kpermass = math.exp(-KP1-KP2*10000*(1/TK-1/TM)/R) #L/g s
    #rhocat = 770 #kg/m3
    #porosity = 0.32
    #kc = 9.7*10**(-5)*SuperVelocity**0.0125 #m/s
    #SAsphere = 4*pi*rSphere**2 #m2
    #Voloftube1sphere = TubeArea*dSphere #m3
    #ac = SAsphere/Voloftube1sphere #m2/m3
    #visEtOH = math.exp(-7.3714+2770/(74.6787+TK)) #mPa s 
    #D = (7.4*10**(-8)*TK*(46.07)**0.5)/(100*100*visEtOH*92.5**0.6) #m2/s
    #De = D*porosity/Tort #m2/s
    #Thiele = rSphere*math.sqrt(kpermass*rhocat/De)
    #eta = (3/(Thiele**2))*(Thiele/math.tanh(Thiele)-1)
    #Omega = eta/(1+eta*kpermass*rhocat/(kc*ac))  
                   
    rate=1*(kpermass*CBA*CEtOH)/((1+KW*CW)**2)
    #rate=1*(kpermass*CBA*CEtOH)/((1+KEtOH*CEtOH+KW*CW)**2)
    
    dCBAdW = -rate/flowLs
    dCEtOHdW = -rate/flowLs
    dCEBdW = rate/flowLs 
    dCWdW = rate/flowLs 
         
    return [dCBAdW, dCEtOHdW, dCEBdW, dCWdW]
    
def GeneratePredictions(Parameters, SingleExpCondition, Mass, Diameter):
    AllPredictions=np.zeros([len(Parameters),4])
    MeasurablePredictions=np.zeros([len(Parameters),2])
    for i in range (0, len(Parameters)):
        AllPredictions[i,:]=odeint(kinetic_model,[SingleExpCondition[2],17.09-1.6824*SingleExpCondition[2],0,0],[0,Mass], mxstep = 3000, args=(SingleExpCondition, Diameter, Parameters[i]))[1,:]
        MeasurablePredictions[i,0]=AllPredictions[i,0]
        MeasurablePredictions[i,1]=AllPredictions[i,2]
    return MeasurablePredictions
    
FactorialPredictedOutletValues=GeneratePredictions(FactorialGeneratedParameters, SingleExpCondition, CatalystMass, CatalystDiameter)
DiscriminationPredictedOutletValues=GeneratePredictions(DiscriminationGeneratedParameters, SingleExpCondition, CatalystMass, CatalystDiameter)
MBDoEPredictedOutletValues=GeneratePredictions(MBDoEGeneratedParameters, SingleExpCondition, CatalystMass, CatalystDiameter)

#Plot Histograms
#plt.figure(2)
bins= np.linspace(0,1.6,1000)
plt.hist(FactorialPredictedOutletValues[:,0], bins, normed=True, color='r', alpha=0.5, label='After Factorial')
plt.hist(DiscriminationPredictedOutletValues[:,0], bins, normed=True, color='b', alpha=0.5, label='After Model Discrimination')
plt.hist(MBDoEPredictedOutletValues[:,0], bins, normed=True, color='g', alpha=0.5, label='After MBDoE')
#plt.hist(FactorialPredictedOutletValues[:,1], bins, normed=True, color='c', alpha=0.5, label='Predicted Outlet EB after factorial')
#plt.hist(DiscriminationPredictedOutletValues[:,1], bins, normed=True, color='y', alpha=0.5, label='Predicted Outlet EB after model discrimination')
#plt.hist(MBDoEPredictedOutletValues[:,1], bins, normed=True, color='k', alpha=0.5, label='Predicted Outlet EB after MBDoE')
plt.xlim(0.7, 0.9)
plt.title('100 oC, 20 uL/min, 1.0M')
plt.xlabel('Predicted Benzoic Acid Outlet Concentration')
plt.ylabel('Observed Frequency')
plt.legend(loc='best')
plt.show()

end=time.time()
runtime=end-start
print(runtime)