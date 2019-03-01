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
#import os
#import openpyxl
import time

start=time.time()
#This code only works for 2 measured variables
n_y = 2
Number_variables = 3
Variableranges=np.zeros([Number_variables,2])
Variableranges[0,:]=[80, 140]
Variableranges[1,:]=[7.5, 30]
Variableranges[2,:]=[0.9, 1.55]

Latin_Number_sample = 27
FactorialLevels = [3,3,3]

TrueParameters = [9.1, 8.1]
ParameterGuess= [11, 7]  

Sigma = [0.03, 0.0165]

def LatinGenerator(N_variables, N_sample, VarRanges):
    sampling=pyDOE.lhs(N_variables, N_sample)
    Designs=np.zeros([N_sample,N_variables])
    for i in range (0, N_variables):
        Designs[:,i]=VarRanges[i,0]+sampling[:,i]*(VarRanges[i,1]-VarRanges[i,0])
    return Designs
    
def FactorialGenerator(levels, VarRanges):
    sampling=pyDOE.fullfact(levels) #for each variable you input the level you want
    Designs=np.zeros([len(sampling),len(levels)])
    for i in range (0, len(levels)):
        Designs[:,i]=VarRanges[i,0]+sampling[:,i]*(VarRanges[i,1]-VarRanges[i,0])
    return Designs


LatinExpCond=LatinGenerator(Number_variables, Latin_Number_sample, Variableranges)
FactorialExpCond=FactorialGenerator(FactorialLevels, Variableranges)
#Here you need to pick which design to use
ExpConditions = LatinExpCond


#3D plotting of experimental design space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ExpConditions[:,0], ExpConditions[:,1], ExpConditions[:,2])
ax.set_xlabel('Temperature (oC)')
ax.set_ylabel('Flowrate (uL/min)')
ax.set_zlabel('Inlet Concentration (M)')
plt.show()

def kinetic_model(c,W,u,theta):
    CBA = c[0] #mol/L
    CEtOH = c[1]
    CEB = c[2] #mol/L
    CW = c[3]
    
    tempC = u[0] #oC
    flowuLmin = u[1] #uL/min
    InletC = u[2] #mol/L
    
    KP1 = theta[0]
    KP2 = theta[1]
    R = 8.314

    flowLs = flowuLmin*10**(-6)/60 #L/s
    TK=tempC+273.15 #K
    TM=((140+70)/2)+273.15 #K
    
    kpermass = math.exp(-KP1-KP2*10000*(1/TK-1/TM)/R) #L/g s
    rate=kpermass*CBA
       
    dCBAdW = -rate/flowLs
    dCEtOHdW = -rate/flowLs
    dCEBdW = rate/flowLs 
    dCWdW = rate/flowLs 
         
    return [dCBAdW, dCEtOHdW, dCEBdW, dCWdW]
    
#Function to create model predicted outlet concentrations for a given set of experimental conditions and a given set of Kinetic parameters    
def generatepredictions(ExpCond, theta):
    ConcPredicted=np.zeros([len(ExpCond),2])
    for i in range (0, len(ExpCond)):
        soln = odeint(kinetic_model,[ExpCond[i,2],17.09-1.6824*ExpCond[i,2],0,0],[0,9.8174*10**(-5)], mxstep = 3000, args=(ExpCond[i,:],theta))
        CBA = soln[1, 0]
        CEB = soln[1, 2]
        ConcPredicted[i,:] = [CBA, CEB]
    return ConcPredicted
#SimulatedExpResultsNoError=generatepredictions(ExpConditions, TrueParameters)
#print SimulatedExpResultsNoError
        
def ModelplusError(ExpCond, theta, stdev):
    ModelConc=generatepredictions(ExpCond, theta)
    Error=np.zeros([len(ExpCond),2])
    for i in range (0, 2):
        Error[:,i]=np.random.normal(0,stdev[i],len(ExpCond))
    SimulatedConc=ModelConc+Error
    return SimulatedConc, Error
SimulatedExpResults, err=ModelplusError(ExpConditions, TrueParameters, Sigma)
#print SimulatedExpResults

def loglikelihood(theta, Measurments, ExpCond):
    ModelPredictions=generatepredictions(ExpCond, theta)
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
#TestLog=loglikelihood(ParameterGuess, SimulatedExpResults, ExpConditions)
#print TestLog
    
    #Function to get parameter estimate
def parameter_estimation(thetaguess, measureddata, inputconditions):
    new_estimate = minimize(loglikelihood, thetaguess, method = 'Nelder-Mead', options = {'maxiter':2000}, args=(measureddata, inputconditions,))
    #print "preformed PE"
    return new_estimate  
minimisedlog = parameter_estimation(ParameterGuess, SimulatedExpResults, ExpConditions)
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

def sensitivity(OneExp, epsilon, TrueParams):
    KPMatrix=perturbation(epsilon,TrueParams)
    PredictedValues= np.zeros([len(KPMatrix),4])
    PredictedMeasurable= np.zeros([len(KPMatrix),2])
    for i in range(len(KPMatrix)):
        Solution = odeint(kinetic_model,[OneExp[2], 17.09-1.6824*OneExp[2], 0, 0], [0,9.8174*10**(-5)], mxstep = 3000, args=(OneExp, KPMatrix[i,:]))
        PredictedValues[i,:] = Solution[1,:]
        PredictedMeasurable[i,0]=Solution[1,0]
        PredictedMeasurable[i,1]=Solution[1,2]
    sensitivity_matrix = np.zeros([len(TrueParams),n_y])
    for j in range(len(TrueParams)):
        for k in range(n_y):
            sensitivity_matrix[j,k] = ((PredictedMeasurable[j,k] - PredictedMeasurable[-1,k])/(epsilon*TrueParams[j]))  
    return sensitivity_matrix  
#testsens=sensitivity(Examplexp, Disturbance, params)
#print testsens

#Make information matrix for a single experiment
def information(OneExp,TrueParams, epsilon):
    Fisher=np.zeros([len(TrueParams),len(TrueParams)])
    for j in range(n_y):
        sens=sensitivity(OneExp, epsilon, TrueParams)[:,j]
        Fisher = Fisher + (1/(Sigma[j]**2)) * np.outer(sens,sens)
    #print "Preformed fisher for 1 exp"
    return Fisher
#TestFisher=information(Examplexp, params, Disturbance)   
#print TestFisher

#Here we get Fisher for all N Experiments
def obs_Fisher(ExpCond,epsilon,TrueParams):
    obs_information = np.zeros([len(ExpCond),len(TrueParams),len(TrueParams)])
    for j in range(len(ExpCond)):
        obs_information[j,:,:] = information(ExpCond[j,:], TrueParams, epsilon)
    overall_obs_Fisher = np.zeros([len(TrueParams),len(TrueParams)])
    for j in range(len(ExpCond)):
        overall_obs_Fisher = overall_obs_Fisher + obs_information[j,:,:]
    return overall_obs_Fisher 
#testobsFisher=obs_Fisher(ExpConditions, Disturbance, params)   
#print testobsFisher

def obs_covariance(ExpCond,epsilon,TrueParams):
    obs_variance_matrix = np.linalg.inv(obs_Fisher(ExpCond,epsilon,TrueParams))
    return obs_variance_matrix
Cov=obs_covariance(ExpConditions,Disturbance,params)

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

end=time.time()
runtime=end-start
print ("the run time is %f" %(runtime))
print ("the true param values are")
print TrueParameters
print ("the pramater estiamtes are")
print params
print ("the residuals are %f and the reference value is %f" %(wt_residuals, chisq_ref))
print ("the t values are are") 
print tvalue
print tref
print ("the confidence intervals are") 
print ConfInt
print ("the correlation matrix is")
print Corr
print ("the covariance matrix is")
print Cov