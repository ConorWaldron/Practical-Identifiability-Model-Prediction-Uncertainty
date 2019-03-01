import pyDOE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.integrate import odeint
from scipy.optimize import minimize
import openpyxl
#import pandas as pd
import scipy.stats as stats
#import os
#import openpyxl
import time

start=time.time()


TrueParameters = [9.1, 8.1]
ParameterGuess= [11, 7]  

n_y = 2
Sigma = [0.03, 0.0165]

Number_variables = 3
Variableranges=np.zeros([Number_variables,2])
Variableranges[0,:]=[80, 140]
Variableranges[1,:]=[7.5, 30]
Variableranges[2,:]=[0.9, 1.55]

#Experimental conditions decided by researcher to start
ExpConditions=np.array([[140, 20, 1.5], [120, 10, 1]])

ObjCriteria="D"
NumberofMBDoEExp = 25

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
    return SimulatedConc
SimulatedExpResults=ModelplusError(ExpConditions, TrueParameters, Sigma)
#print SimulatedExpResults

def PE_MBDoE(ExpCond, y_m, thetaguess, stdev, VarRanges, Criteria):
    def loglikelihood(theta, y_m, ExpCond):
        ModelPredictions=generatepredictions(ExpCond, theta)
        BAPredict=ModelPredictions[:,0]
        EBPredict=ModelPredictions[:,1]
        BAmeasured=y_m[:,0]
        EBmeasured=y_m[:,1]
        rho_1 = BAmeasured - BAPredict
        rho_2 = EBmeasured - EBPredict
        rho = (rho_1/stdev[0])**2+(rho_2/stdev[1])**2
        residuals = np.sum(rho)
        neg_loglikelihood = math.log(2*math.pi) + 0.5 * (math.log(stdev[0]**2) + math.log(stdev[1]**2)) + 0.5 * residuals
        obj_fun = 0.5 * residuals
        return obj_fun
    #CheckLog=loglikelihood(ExpCond, y_m, theta, stdev)
    
    #Function to get parameter estimate
    def parameter_estimation(theta, y_m, ExpCond):
        new_estimate = minimize(loglikelihood, theta, method = 'Nelder-Mead', options = {'maxiter':2000}, args=(y_m, ExpCond,))
        #print "preformed minimisation of log liklihood"
        return new_estimate
    minimisedlog = parameter_estimation(thetaguess, y_m, ExpCond)
    params = minimisedlog.x
    wt_residuals = 2*minimisedlog.fun
    
        ## Adequacy test/chisquare #
    alpha = 0.05
    confidence_level = 1 - alpha
    dofreedom = (((len(ExpCond)) * n_y) - len(thetaguess))
    def chisquare_test(conf_level, dof):
        ref_chisquare = stats.chi2.ppf((conf_level),dof)
        return ref_chisquare
    chisq_ref = chisquare_test(confidence_level, dofreedom)
    
     ##Create matrix of perturbed parameters to be used later for making information matrix
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
                sensitivity_matrix[j,k] = ((PredictedMeasurable[j,k] - PredictedMeasurable[-1,k])/(epsilon*TrueParams[j]))  #divide by eepslion*theta?
        return sensitivity_matrix 
    
    def information(OneExp,TrueParams, epsilon):
        Fisher=np.zeros([len(TrueParams),len(TrueParams)])
        for j in range(n_y):
            sens=sensitivity(OneExp, epsilon, TrueParams)[:,j]
            Fisher = Fisher + (1/(stdev[j]**2)) * np.outer(sens,sens)
        return Fisher
    #testFisher=information(Examplexp, params, Disturbance)    #Why do I have negative information here!!!
    
    #Here we get Fisher for all N Experiments
    def obs_Fisher(ExpCond,epsilon,TrueParams):
        obs_information = np.zeros([len(ExpCond),len(TrueParams),len(TrueParams)])
        for j in range(len(ExpCond)):
            obs_information[j,:,:] = information(ExpCond[j,:], TrueParams, epsilon)
        overall_obs_Fisher = np.zeros([len(TrueParams),len(TrueParams)])
        for j in range(len(ExpCond)):
            overall_obs_Fisher = overall_obs_Fisher + obs_information[j,:,:]
        return overall_obs_Fisher 
    
    def obs_covariance(ExpCond,epsilon,TrueParams):
        obs_variance_matrix = np.linalg.inv(obs_Fisher(ExpCond,epsilon,TrueParams))
        return obs_variance_matrix
    Cov=obs_covariance(ExpCond,Disturbance,params)
    
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
    Cint, tVal, tRef=t_test(Cov, params,confidence_level,dofreedom)
    
    #expected Fisher with 1 new experiment
    def ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon):
        PastFisher=np.linalg.inv(PastCovariance)
        FisherofDesign=information(NewExp, TrueParams, epsilon)
        TotalExpectedFisher=PastFisher+FisherofDesign
        ExpectedCov=np.linalg.inv(TotalExpectedFisher)
        return ExpectedCov
    #TestGuessDesign=[111, 17, 1.25]
    #TestExpCov=ExpectedCovFunction(Cov, TestGuessDesign, params, Disturbance)
    
    def ObjFunction(NewExp, PastCovariance, TrueParams, epsilon, Criteria):
        if Criteria == "A":
            Obj=np.trace(ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon))
        elif Criteria =="D":
            Obj=np.log(np.linalg.det(ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon))) #we take log of determinant
        elif Criteria ==1:
            Obj=ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon)[0,0]
        elif Criteria ==2:
            Obj=ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon)[1,1]    
        else:
            e,v=np.linalg.eig(ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon))    
            Obj=np.max(e)
        return Obj
    
    #need to set a seed so I get the same results everytime so the work is reproducible
    def LatinGenerator(N_variables, N_sample, VarRanges):
        np.random.seed(1) #This sets the seed for my random functions like the latin generator to make them give the same results everytime.
        sampling=pyDOE.lhs(N_variables, N_sample)
        Designs=np.zeros([N_sample,N_variables])
        for i in range (0, N_variables):
            Designs[:,i]=VarRanges[i,0]+sampling[:,i]*(VarRanges[i,1]-VarRanges[i,0])
        return Designs
    NumberofExpInScreen = 10000
    Screening=LatinGenerator(Number_variables,NumberofExpInScreen,VarRanges)
    
    def FindBestGuess(Designs, PastCovariance, TrueParams, epsilon, Criteria):
        BestObjective=100
        BestDesign=0
        for i in range (0, len(Designs)): 
            CriteriaValue=ObjFunction(Designs[i,:], PastCovariance, TrueParams, epsilon, Criteria)
            if CriteriaValue < BestObjective:
                #print "We found a better guess"
                BestObjective = CriteriaValue
                BestDesign = i
        BestGuessDesign=Designs[BestDesign,:]
        return BestGuessDesign        
    BestDesignFromScreening=FindBestGuess(Screening, Cov, params, Disturbance, Criteria)
    
    def MBDoE(NewExp, PastCovariance, TrueParams, epsilon, Criteria):
        new_design=minimize(ObjFunction, NewExp, method = 'SLSQP', bounds = ([VarRanges[0,0],VarRanges[0,1]],[VarRanges[1,0],VarRanges[1,1]],[VarRanges[2,0],VarRanges[2,1]]), options = {'maxiter':10000, 'ftol':1e-20}, args = (PastCovariance, TrueParams, epsilon, Criteria,))
        return new_design    
    MinimisedMBDoE=MBDoE(BestDesignFromScreening, Cov, params, Disturbance, Criteria)
    NewDesign=MinimisedMBDoE.x
    Objvalue=MinimisedMBDoE.fun
    
    return params, wt_residuals, chisq_ref, Cov, Corr[0,1], Cint, tVal, tRef, NewDesign
    

New_params, New_wt_residuals, New_chisq_ref, New_Cov, New_Corr, New_Cint, New_tVal, New_tRef, New_NewDesign=PE_MBDoE(ExpConditions, SimulatedExpResults, ParameterGuess, Sigma, Variableranges, ObjCriteria)

#Now I need to update the expconditions and result files
After2exp_ExpConditions=np.vstack((ExpConditions, New_NewDesign))

#I want to keep the same results from before, not generate new random ones every time. So just take the bottom row out
#print SimulatedExpResults
WarningResult=ModelplusError(After2exp_ExpConditions, TrueParameters, Sigma)
#print WarningResult
EndofWarning=WarningResult[-1,:]
#print EndofWarning
After2exp_SimulatedExpResults=np.vstack((SimulatedExpResults, EndofWarning))
#print New_SimulatedExpResults

After2exp_ListPE = New_params
After2exp_ListX2 = New_wt_residuals
After2exp_ListX2ref = New_chisq_ref
After2exp_ListCorr = New_Corr
After2exp_ListConfInt = New_Cint
After2exp_ListT = New_tVal
After2exp_ListTref = New_tRef


#Now I need to repeat the PE MBDoE process as many times as I want
def Repeat_PE_MBDoE(Exp, SimResults, thetaguess, stdev, VarRanges, Criteria, ListofPE, ListofX2, ListofX2ref, ListofCorr, ListofConfInt, ListofT, ListofTref):
    
    New_KP, New_X, New_X_ref, New_Covar, New_Correlation, New_CFint, New_tV, New_tR, New_MBDoEExp=PE_MBDoE(Exp, SimResults, thetaguess, stdev, VarRanges, Criteria)

    
    New_ExpConditions=np.vstack((Exp, New_MBDoEExp))
    
    #I want to keep the same results from before, not generate new random ones every time. So just take the bottom row out
    FakeResult=ModelplusError(New_ExpConditions, TrueParameters, Sigma)
    #print WarningResult
    EndofFake=FakeResult[-1,:]
    #print EndofWarning
    New_SimulatedExpResults=np.vstack((SimResults, EndofFake))
    #print New_SimulatedExpResults 
    
    New_ListofPE = np.vstack((ListofPE,New_KP))
    New_ListX2 = np.vstack((ListofX2,New_X))
    New_ListX2ref = np.vstack((ListofX2ref,New_X_ref))
    New_ListofCorr = np.vstack((ListofCorr,New_Correlation))
    New_ListofConfInt = np.vstack((ListofConfInt,New_CFint))
    New_ListofT = np.vstack((ListofT,New_tV))
    New_ListofTref = np.vstack((ListofTref,New_tR))
    
    return New_ExpConditions, New_SimulatedExpResults, New_ListofPE, New_ListX2, New_ListX2ref, New_ListofCorr, New_ListofConfInt, New_ListofT, New_ListofTref
#Test=Repeat_PE_MBDoE(After2exp_ExpConditions, After2exp_SimulatedExpResults, ParameterGuess, Sigma, Variableranges, ObjCriteria, After2exp_ListPE, After2exp_ListX2, After2exp_ListX2ref, After2exp_ListCorr, After2exp_ListConfInt, After2exp_ListT, After2exp_ListTref)

    
def NTimesMBDoE(NumberOfDesigns, ExpConditionsIn, SimulatedExpResultsIn, thetaguess, stdev, VarRanges, Criteria, ListofPEIn, ListofX2In, ListofX2refIn, ListofCorrIn, ListofConfIntIn, ListofTIn, ListofTrefIn):
    MBDoE_ExpConditions=ExpConditionsIn
    MBDoE_SimulatedExpResults=SimulatedExpResultsIn
    ListPE=ListofPEIn
    ListX2=ListofX2In
    ListX2ref=ListofX2refIn
    ListCorr=ListofCorrIn
    ListConfInt=ListofConfIntIn
    ListT=ListofTIn
    ListTref=ListofTrefIn
    
    for i in range(NumberOfDesigns):
        MBDoE_ExpConditions, MBDoE_SimulatedExpResults, ListPE, ListX2, ListX2ref, ListCorr, ListConfInt, ListT, ListTref=Repeat_PE_MBDoE(MBDoE_ExpConditions, MBDoE_SimulatedExpResults, thetaguess, stdev, VarRanges, Criteria, ListPE, ListX2, ListX2ref, ListCorr, ListConfInt, ListT, ListTref)

    return MBDoE_ExpConditions, MBDoE_SimulatedExpResults, ListPE, ListX2, ListX2ref, ListCorr, ListConfInt, ListT, ListTref

FinalValues=NTimesMBDoE(NumberofMBDoEExp, After2exp_ExpConditions, After2exp_SimulatedExpResults, ParameterGuess, Sigma, Variableranges, ObjCriteria, After2exp_ListPE, After2exp_ListX2, After2exp_ListX2ref, After2exp_ListCorr, After2exp_ListConfInt, After2exp_ListT, After2exp_ListTref)

FinalExpList=FinalValues[0]
FinalSimResults=FinalValues[1]
FinalPE=FinalValues[2]
FinalX2=FinalValues[3]
FinalX2ref=FinalValues[4]
FinalCorr=FinalValues[5]
FinalConfInt=FinalValues[6]
Finaltvalues=FinalValues[7]
Finaltref=FinalValues[8]


#append results to a new file
    
wbwrite=openpyxl.Workbook()
wswrite=wbwrite.active #opens the first sheet in the wb
    
#Mainipulating Files
wswrite['A'+str(1)]="Temperature"
wswrite['B'+str(1)]="Flowrate" 
wswrite['C'+str(1)]="Concentration" 
wswrite['D'+str(1)]="BA" 
wswrite['E'+str(1)]="EB"  

wswrite['H'+str(1)]="KP1" 
wswrite['I'+str(1)]="KP2"
#wswrite['J'+str(1)]="KP3" 
wswrite['K'+str(1)]="X2" 
wswrite['L'+str(1)]="X2ref" 
wswrite['M'+str(1)]="KP1 tvalue" 
wswrite['N'+str(1)]="KP2 tvalue"
#wswrite['O'+str(1)]="KP3 tvalue" 
wswrite['P'+str(1)]="tref"
wswrite['Q'+str(1)]="KP1 CInt" 
wswrite['R'+str(1)]="KP2 CInt"
#wswrite['S'+str(1)]="KP3 CInt" 


for i in range (0, len(FinalExpList)-1):
    wswrite['A'+str(i+2)]=FinalExpList[i,0] #using the excel numbering system    
    wswrite['B'+str(i+2)]=FinalExpList[i,1] #using the excel numbering system 
    wswrite['C'+str(i+2)]=FinalExpList[i,2] #using the excel numbering system 
    wswrite['D'+str(i+2)]=FinalSimResults[i,0] #using the excel numbering system 
    wswrite['E'+str(i+2)]=FinalSimResults[i,1] #using the excel numbering system 

for i in range (0, len(FinalX2)):
    wswrite['H'+str(i+3)]=FinalPE[i,0] #using the excel numbering system    
    wswrite['I'+str(i+3)]=FinalPE[i,1] #using the excel numbering system 
    ##wswrite['J'+str(i+3)]=FinalPE[i,2] #using the excel numbering system 
    #wswrite['K'+str(i+3)]=FinalX2[i] #using the excel numbering system 
    #wswrite['L'+str(i+3)]=FinalX2ref[i] #using the excel numbering system 
    wswrite['M'+str(i+3)]=Finaltvalues[i,0] #using the excel numbering system 
    wswrite['N'+str(i+3)]=Finaltvalues[i,1] #using the excel numbering system
    ##wswrite['O'+str(i+3)]=Finaltvalues[i,2] #using the excel numbering system
    #wswrite['P'+str(i+3)]=Finaltref[i] #using the excel numbering system
    wswrite['Q'+str(i+3)]=FinalConfInt[i,0] #using the excel numbering system 
    wswrite['R'+str(i+3)]=FinalConfInt[i,1] #using the excel numbering system
    ##wswrite['S'+str(i+3)]=FinalConfInt[i,2] #using the excel numbering system
    
#Saving File
wbwrite.save("SilicoMBDoESA.xlsx")# overwrites without warning. So be careful

#3D plotting of experimental design space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(FinalExpList[:,0], FinalExpList[:,1], FinalExpList[:,2])
ax.set_xlabel('Temperature (oC)')
ax.set_ylabel('Flowrate (uL/min)')
ax.set_zlabel('Inlet Concentration (M)')
plt.show()

end=time.time()
runtime=end-start
print(runtime)