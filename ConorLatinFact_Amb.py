import pyDOE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.integrate import odeint
from scipy.optimize import minimize
#from openpyxl import Workbook, load_workbook
#import pandas as pd
import scipy.stats as stats
import os
import openpyxl
import time

start=time.time()

os.chdir('C:\Users\Conor\OneDrive - University College London\Closed Loop System\Closed Loop Python')

#This code only works for 2 measured variables
n_y = 2
Number_variables = 3
Variableranges=np.zeros([Number_variables,2])
Variableranges[0,:]=[80, 120]
Variableranges[1,:]=[15, 60]
Variableranges[2,:]=[0.9, 1.55]

Latin_Number_sample = 64
FactorialLevels = [4,4,4]

diametersphere=825
CatMass = 0.1

#First order without mass transfer
#TrueParameters = [16.7, 5.93]
#ParameterGuess= [14, 7] 

#LH 3 param without mass transfer
#TrueParameters = [16.6, 6.41, 0.28]
#ParameterGuess= [14, 8, 1]  

#LH 4 param without mass transfer
#TrueParameters = [15.3, 6.57, 0.81, 0.06]
#ParameterGuess= [14, 8, 1, 0.1]  

#LH 6 param without mass transfer
TrueParameters = [13.4, 6.57, 0.80, 0.24, 0.15, 1.45]
ParameterGuess= [15, 7, 0.5, 0.5, 0.5, 0.5]  

Sigma = [0.03, 0.0165]

def LatinGenerator(N_variables, N_sample, VarRanges):
    sampling=pyDOE.lhs(N_variables, N_sample)
    Designs=np.zeros([N_sample,N_variables])
    for i in range (0, N_variables):
        Designs[:,i]=VarRanges[i,0]+sampling[:,i]*(VarRanges[i,1]-VarRanges[i,0])
    return Designs
    
def FactorialGenerator(levels, VarRanges):
    sampling=pyDOE.fullfact(levels) #for each variable you input the level you want
    for i in range (0, Number_variables):
        sampling[:,i]=sampling[:,i]/(levels[i]-1)
    Designs=np.zeros([len(sampling),len(levels)])
    for i in range (0, len(levels)):
        Designs[:,i]=VarRanges[i,0]+sampling[:,i]*(VarRanges[i,1]-VarRanges[i,0])
    return Designs


LatinExpCond=LatinGenerator(Number_variables, Latin_Number_sample, Variableranges)
FactorialExpCond=FactorialGenerator(FactorialLevels, Variableranges)
#Here you need to pick which design to use
ExpConditions = FactorialExpCond
DspheresList=np.zeros(len(ExpConditions))
CatMassesList=np.zeros(len(ExpConditions))
for i in range (0, len(ExpConditions)):
    DspheresList[i]=diametersphere
    CatMassesList[i]=CatMass

#3D plotting of experimental design space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ExpConditions[:,0], ExpConditions[:,1], ExpConditions[:,2])
ax.set_xlabel('Temperature (oC)')
ax.set_ylabel('Flowrate (uL/min)')
ax.set_zlabel('Inlet Concentration (M)')
plt.show()

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
    KEtOH  = theta[3]
    KBA  = theta[4]
    KEB  = theta[5]
    R = 8.314
    pi = math.pi
    Tort = 3.745
    
    flowLs = flowuLmin*10**(-6)/60 #L/s
    TK=tempC+273.15 #K
    TM=((140+70)/2)+273.15 #K
    rSphere = dSphere/2 #m
    
    TubeDiameter = 1000*10**(-6) #m
    TubeArea = pi*(TubeDiameter/2)**(2) #m2
    SuperVelocity = (flowLs/1000)/TubeArea #m/s
    
    kpermass = math.exp(-KP1-KP2*10000*(1/TK-1/TM)/R) #L/g s
    rhocat = 770 #kg/m3
    porosity = 0.32
    kc = 9.7*10**(-5)*SuperVelocity**0.0125 #m/s
    SAsphere = 4*pi*rSphere**2 #m2
    Voloftube1sphere = TubeArea*dSphere #m3
    ac = SAsphere/Voloftube1sphere #m2/m3
    visEtOH = math.exp(-7.3714+2770/(74.6787+TK)) #mPa s 
    D = (7.4*10**(-8)*TK*(46.07)**0.5)/(100*100*visEtOH*92.5**0.6) #m2/s
    De = D*porosity/Tort #m2/s
    Thiele = rSphere*math.sqrt(kpermass*rhocat/De)
    eta = (3/(Thiele**2))*(Thiele/math.tanh(Thiele)-1)
    Omega = eta/(1+eta*kpermass*rhocat/(kc*ac))  

    #rate=1*(kpermass*CBA*CEtOH)
    #rate=Omega*(kpermass*CBA*CEtOH)               
    #rate=1*(kpermass*CBA*CEtOH)/((1+KW*CW)**2)
    #rate=1*(kpermass*CBA*CEtOH)/((1+KEtOH*CEtOH+KW*CW)**2)
    rate=1*(kpermass*CBA*CEtOH)/((1+KEtOH*CEtOH+KW*CW+KBA*CBA+KEB*CEB)**2)
        
    dCBAdW = -rate/flowLs
    dCEtOHdW = -rate/flowLs
    dCEBdW = rate/flowLs 
    dCWdW = rate/flowLs 
         
    return [dCBAdW, dCEtOHdW, dCEBdW, dCWdW]
    
#Function to create model predicted outlet concentrations for a given set of experimental conditions and a given set of Kinetic parameters    
def generatepredictions(ExpCond, theta, dsphereum, catmassgram):
    ConcPredicted=np.zeros([len(ExpCond),2])
    for i in range (0, len(ExpCond)):
        soln = odeint(kinetic_model,[ExpCond[i,2],17.09-1.6824*ExpCond[i,2],0,0],[0,catmassgram], mxstep = 3000, args=(ExpCond[i,:], dsphereum, theta))
        CBA = soln[1, 0]
        CEB = soln[1, 2]
        ConcPredicted[i,:] = [CBA, CEB]
    return ConcPredicted
#SimulatedExpResultsNoError=generatepredictions(ExpConditions, TrueParameters, diametersphere, CatMass)
        
def ModelplusError(ExpCond, theta, dsphereum, catmassgram, stdev):
    ModelConc=generatepredictions(ExpCond, theta, dsphereum, catmassgram)
    Error=np.zeros([len(ExpCond),2])
    for i in range (0, 2):
        Error[:,i]=np.random.normal(0,stdev[i],len(ExpCond))
    SimulatedConc=ModelConc+Error
    return SimulatedConc
SimulatedExpResults=ModelplusError(ExpConditions, TrueParameters, diametersphere, CatMass, Sigma)

def loglikelihood(theta, Measurments, ExpCond, dsphereum, catmassgram):
    ModelPredictions=generatepredictions(ExpCond, theta, dsphereum, catmassgram)
    BAPredict=ModelPredictions[:,0]
    EBPredict=ModelPredictions[:,1]
    BAmeasured=Measurments[:,0]
    EBmeasured=Measurments[:,1]
    rho_1 = BAmeasured - BAPredict
    rho_2 = EBmeasured - EBPredict
    rho = (rho_1/Sigma[0])**2+(rho_2/Sigma[1])**2
    residuals = np.sum(rho)
    neg_loglikelihood = math.log(2*math.pi) + 0.5 * (math.log(Sigma[0]**2) + math.log(Sigma[1]**2)) + 0.5 * residuals
    obj_fun = 0.5 * residuals
    return obj_fun
#TestLog=loglikelihood(ParameterGuess, SimulatedExpResults, ExpConditions, diametersphere, CatMass)
    
    #Function to get parameter estimate
def parameter_estimation(thetaguess, measureddata, inputconditions, dsphereum, catmassgram):
    new_estimate = minimize(loglikelihood, thetaguess, method = 'Nelder-Mead', options = {'maxiter':2000}, args=(measureddata, inputconditions, dsphereum, catmassgram,))
    #print "preformed PE"
    return new_estimate  
minimisedlog = parameter_estimation(ParameterGuess, SimulatedExpResults, ExpConditions, diametersphere, CatMass)
params = minimisedlog.x
wt_residuals = 2*minimisedlog.fun

## Adequacy test/chisquare #
alpha = 0.05
confidence_level = 1 - alpha
dofreedom = (((len(ExpConditions)) * 2) - len(ParameterGuess))
chisq_ref = stats.chi2.ppf((confidence_level),dofreedom)

#Create matrix of perturbed parameters to be used later for making information matrix
Disturbance = 0.01
def perturbation(epsilon,TrueParams):      
    perturbated_matrix = np.zeros([len(TrueParams)+1,len(TrueParams)])
    for j in range(len(TrueParams)):
        for k in range(len(TrueParams)):
            if j==k:
                perturbated_matrix[j,k] = TrueParams[j] * (1 + epsilon)
            else:
                perturbated_matrix[j,k] = TrueParams[k]
    for j in range(len(TrueParams)):
        perturbated_matrix[-1,j] = TrueParams[j]
    return perturbated_matrix
#PerturbedParameterMatrix=perturbation(Disturbance,params)

Examplexp=np.zeros(3)
Examplexp[0]=120
Examplexp[1]=20
Examplexp[2]=1.5
ExampleDsphere=825
ExampleCatMass=0.0982
def sensitivity(OneExp, epsilon, TrueParams, dsphereum, catmassgram):
    KPMatrix=perturbation(epsilon,TrueParams)
    PredictedValues= np.zeros([len(KPMatrix),4])
    PredictedMeasurable= np.zeros([len(KPMatrix),2])
    for i in range(len(KPMatrix)):
        Solution = odeint(kinetic_model,[OneExp[2], 17.09-1.6824*OneExp[2], 0, 0], [0,catmassgram], mxstep = 3000, args=(OneExp, dsphereum, KPMatrix[i,:]))
        PredictedValues[i,:] = Solution[1,:]
        PredictedMeasurable[i,0]=Solution[1,0]
        PredictedMeasurable[i,1]=Solution[1,2]
    sensitivity_matrix = np.zeros([len(TrueParams),n_y])
    for j in range(len(TrueParams)):
        for k in range(n_y):
            sensitivity_matrix[j,k] = ((PredictedMeasurable[j,k] - PredictedMeasurable[-1,k])/(epsilon*TrueParams[j]))  
    return sensitivity_matrix  
#testsens=sensitivity(Examplexp, Disturbance, params, ExampleDsphere, ExampleCatMass)

#Make information matrix for a single experiment
def information(OneExp,TrueParams, epsilon,dsphereum, catmassgram):
    Fisher=np.zeros([len(TrueParams),len(TrueParams)])
    for j in range(n_y):
        sens=sensitivity(OneExp, epsilon, TrueParams, dsphereum, catmassgram)[:,j]
        Fisher = Fisher + (1/(Sigma[j]**2)) * np.outer(sens,sens)
    #print "Preformed fisher for 1 exp"
    return Fisher
#testFisher=information(Examplexp, params, Disturbance, ExampleDsphere, ExampleCatMass)   

#Here we get Fisher for all N Experiments
def obs_Fisher(ExpCond,epsilon,TrueParams, dsphereumvector, catmassgramvector):
    obs_information = np.zeros([len(ExpCond),len(TrueParams),len(TrueParams)])
    for j in range(len(ExpCond)):
        obs_information[j,:,:] = information(ExpCond[j,:], TrueParams, epsilon, dsphereumvector[j], catmassgramvector[j])
    overall_obs_Fisher = np.zeros([len(TrueParams),len(TrueParams)])
    for j in range(len(ExpCond)):
        overall_obs_Fisher = overall_obs_Fisher + obs_information[j,:,:]
    return overall_obs_Fisher 
#testobsFisher=obs_Fisher(ExpConditions, Disturbance, params, DspheresList, CatMassesList)   

def obs_covariance(ExpCond,epsilon,TrueParams, dsphereumvector, catmassgramvector):
    obs_variance_matrix = np.linalg.inv(obs_Fisher(ExpCond,epsilon,TrueParams, dsphereumvector, catmassgramvector))
    return obs_variance_matrix
Cov=obs_covariance(ExpConditions,Disturbance,params, DspheresList,CatMassesList)

def correlation(Covariance):
    correlationmatrix = np.zeros([len(Covariance),len(Covariance)])
    for i in range(len(Covariance)):
        for j in range(len(Covariance)):
            correlationmatrix[i,j] = Covariance[i,j]/(np.sqrt(Covariance[i,i] * Covariance[j,j]))
    return correlationmatrix
Corr=correlation(Cov)

def t_test(Covariance, TrueParams, conf_level, dof):  
    t_values = np.zeros(len(TrueParams))
    conf_interval = np.zeros(len(TrueParams))
    for j in range(len(TrueParams)):
        conf_interval[j] = np.sqrt(Covariance[j,j]) * stats.t.ppf((1 - ((1-conf_level)/2)), dof) 
        t_values[j] = TrueParams[j]/(conf_interval[j])
    t_ref = stats.t.ppf((1-(1-conf_level)),dof)
    return conf_interval,t_values,t_ref  
ttests=t_test(Cov, params,confidence_level,dofreedom)
ConfInt=ttests[0]
tvalue=ttests[1]
tref=ttests[2]

#append results to a new file
    
wbwrite=openpyxl.Workbook()
wswrite=wbwrite.active #opens the first sheet in the wb
    
#Mainipulating Files
wswrite['A'+str(1)]="Temperature"
wswrite['B'+str(1)]="Flowrate" 
wswrite['C'+str(1)]="Concentration" 
wswrite['D'+str(1)]="dsphere" 
wswrite['E'+str(1)]="cat mass"
wswrite['F'+str(1)]="BAC" 
wswrite['G'+str(1)]="EBC"  

#wswrite['H'+str(64)]="KP1" 
#wswrite['I'+str(64)]="KP2"
#wswrite['J'+str(64)]="KP3" 
#wswrite['K'+str(64)]="KP4" 
#wswrite['L'+str(64)]="X2" 
#wswrite['M'+str(64)]="X2 ref" 
#wswrite['N'+str(64)]="KP1 tvalue"
#wswrite['O'+str(64)]="KP2 tvalue" 
#wswrite['P'+str(64)]="KP3 tvalue"
#wswrite['Q'+str(64)]="KP4 tvalue" 
#wswrite['R'+str(64)]="tref"
#wswrite['S'+str(64)]="KP1 CInt"
#wswrite['T'+str(64)]="KP2 CInt" 
#wswrite['U'+str(64)]="KP3 CInt"
#wswrite['V'+str(64)]="KP4 CInt"

#wswrite['H'+str(65)]=params[0] 
#wswrite['I'+str(65)]=params[1]
#wswrite['J'+str(65)]=params[2]
#wswrite['K'+str(65)]=params[3]
#wswrite['L'+str(65)]=wt_residuals
#wswrite['M'+str(65)]=chisq_ref 
#wswrite['N'+str(65)]=tvalue[0]
#wswrite['O'+str(65)]=tvalue[1] 
#wswrite['P'+str(65)]=tvalue[2]
#wswrite['Q'+str(65)]=tvalue[3] 
#wswrite['R'+str(65)]=tref
#wswrite['S'+str(65)]=ConfInt[0] 
#wswrite['T'+str(65)]=ConfInt[1]
#wswrite['U'+str(65)]=ConfInt[2]
#wswrite['V'+str(65)]=ConfInt[3]

for i in range (0, len(ExpConditions)):
    wswrite['A'+str(i+2)]=ExpConditions[i,0] #using the excel numbering system    
    wswrite['B'+str(i+2)]=ExpConditions[i,1] #using the excel numbering system 
    wswrite['C'+str(i+2)]=ExpConditions[i,2] #using the excel numbering system 
    wswrite['D'+str(i+2)]=diametersphere #using the excel numbering system 
    wswrite['E'+str(i+2)]=CatMass #using the excel numbering system 
    wswrite['F'+str(i+2)]=SimulatedExpResults[i,0] #using the excel numbering system 
    wswrite['G'+str(i+2)]=SimulatedExpResults[i,1] #using the excel numbering system

#Saving File
wbwrite.save("Final_Silico_6ParamModel.xlsx")# overwrites without warning. So be careful

end=time.time()
runtime=end-start
print ("the run time is %f" %(runtime))
print ("the true param values are ")
print TrueParameters
print ("the pramater estiamtes are ")
print params
print ("the residuals are %f and the reference value is %f" %(wt_residuals, chisq_ref))
print ("the t values are are ")
print tvalue
print tref
print ("the confidence intervals are ")
print ConfInt
print ("the correlation matrix is")
print Corr
print ("the covariance matrix is")
print Cov