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
import os
#import openpyxl
import time

start=time.time()
os.chdir('C:\Users\Conor\OneDrive - University College London\Closed Loop System\Closed Loop Python')

TrueParameters = [16.66, 6.88, 0.53]
ParameterGuess= [14, 8, 1]   

#TrueParameters = [15.3, 6.57, 0.81, 0.06]
#ParameterGuess= [15, 7, 1, 0.05]   

n_y = 2
Sigma = [0.03, 0.0165]

Number_variables = 3
Variableranges=np.zeros([Number_variables,2])
Variableranges[0,:]=[80, 120]
Variableranges[1,:]=[15, 60]
Variableranges[2,:]=[0.9, 1.55]

diametersphere=950
CatMass = 0.1156

#Experimental conditions decided by researcher to start
ExpConditions=np.array([[120, 40, 1.5], [120, 40, 1], [120, 20, 1.5], [120, 20, 1.0], [100, 40, 1.5], [100, 40, 1.0], [100, 20, 1.5], [100, 20, 1]])
RealExpResultsFactorial1=np.array([[1.110687869, 0.380589449],[0.749128264, 0.2727700385], [0.919522421,0.629868695], [0.601725525,0.447566344], [1.353002808,0.1627679025], [0.900508501,0.1110676215], [1.24497604,0.2883883965], [0.818609346,0.202949932] ])
ObjCriteria="E"
NumberofMBDoEExp = 2 #56 

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
                   
    rate=1*(kpermass*CBA*CEtOH)/((1+KW*CW)**2)
    #rate=1*(kpermass*CBA*CEtOH)/((1+KEtOH*CEtOH+KW*CW)**2)
    
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
SimulatedExpResultsNoError=generatepredictions(ExpConditions, TrueParameters, diametersphere, CatMass)
#print SimulatedExpResultsNoError
        
def ModelplusError(ExpCond, theta, dsphereum, catmassgram, stdev):
    ModelConc=generatepredictions(ExpCond, theta, dsphereum, catmassgram)
    Error=np.zeros([len(ExpCond),2])
    for i in range (0, 2):
        Error[:,i]=np.random.normal(0,stdev[i],len(ExpCond))
    SimulatedConc=ModelConc+Error
    return SimulatedConc
#SimulatedExpResults=ModelplusError(ExpConditions, TrueParameters, diametersphere, CatMass, Sigma) #Here the problem is we are simulating experimental results when we should be using the real ones
SimulatedExpResults=RealExpResultsFactorial1 #We have real results we do not need to simulate
#print SimulatedExpResults

def PE_MBDoE(ExpCond, y_m, thetaguess, stdev, VarRanges, Criteria, dsphereum, catmassgram):
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
    #TestLog=loglikelihood(thetaguess, y_m, ExpCond, diametersphere, CatMass)
  
    #Function to get parameter estimate
    def parameter_estimation(thetaguess, measureddata, inputconditions, dsphereum, catmassgram):
        new_estimate = minimize(loglikelihood, thetaguess, method = 'Nelder-Mead', options = {'maxiter':2000}, args=(measureddata, inputconditions, dsphereum, catmassgram,))
        #print "preformed PE"
        return new_estimate  
    minimisedlog = parameter_estimation(ParameterGuess, y_m, ExpCond, dsphereum, catmassgram)
    params = minimisedlog.x
    #print params
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
    def obs_Fisher(ExpCond,epsilon,TrueParams, dsphereum, catmassgram):
        obs_information = np.zeros([len(ExpCond),len(TrueParams),len(TrueParams)])
        for j in range(len(ExpCond)):
            obs_information[j,:,:] = information(ExpCond[j,:], TrueParams, epsilon, dsphereum, catmassgram)
        overall_obs_Fisher = np.zeros([len(TrueParams),len(TrueParams)])
        for j in range(len(ExpCond)):
            overall_obs_Fisher = overall_obs_Fisher + obs_information[j,:,:]
        return overall_obs_Fisher 
    #testobsFisher=obs_Fisher(ExpConditions, Disturbance, params, dsphereum, catmassgram)   

    def obs_covariance(ExpCond,epsilon,TrueParams, dsphereum, catmassgram):
        obs_variance_matrix = np.linalg.inv(obs_Fisher(ExpCond,epsilon,TrueParams, dsphereum, catmassgram))
        return obs_variance_matrix
    Cov=obs_covariance(ExpCond,Disturbance,params, dsphereum, catmassgram)
    #print Cov
    
    def correlation(Covariance):
        correlationmatrix = np.zeros([len(Covariance),len(Covariance)])
        for i in range(len(Covariance)):
            for j in range(len(Covariance)):
                correlationmatrix[i,j] = Covariance[i,j]/(np.sqrt(Covariance[i,i] * Covariance[j,j]))
        return correlationmatrix
    Corr=correlation(Cov)
    #print Corr
    
    def t_test(Covariance, TrueParams, conf_level, dof):  
        t_values = np.zeros(len(TrueParams))
        conf_interval = np.zeros(len(TrueParams))
        for j in range(len(TrueParams)):
            conf_interval[j] = np.sqrt(Covariance[j,j]) * stats.t.ppf((1 - ((1-conf_level)/2)), dof) 
            t_values[j] = TrueParams[j]/(conf_interval[j])
        t_ref = stats.t.ppf((1-(1-conf_level)),dof)
        return conf_interval,t_values,t_ref  
    Cint, tVal, tRef=t_test(Cov, params,confidence_level,dofreedom)
    #print Cint
    #print tVal
    #print tRef
    
    #expected Fisher with 1 new experiment
    def ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon, dsphereum, catmassgram):
        PastFisher=np.linalg.inv(PastCovariance)
        FisherofDesign=information(NewExp, TrueParams, epsilon, dsphereum, catmassgram)
        TotalExpectedFisher=PastFisher+FisherofDesign
        ExpectedCov=np.linalg.inv(TotalExpectedFisher)
        return ExpectedCov
    #TestGuessDesign=[111, 17, 1.25]
    #TestExpCov=ExpectedCovFunction(Cov, TestGuessDesign, params, Disturbance, dsphereum, catmassgram)
    #print TestExpCov
    
    def ObjFunction(NewExp, PastCovariance, TrueParams, epsilon, Criteria, dsphereum, catmassgram):
        if Criteria == "A":
            Obj=np.trace(ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon, dsphereum, catmassgram))
        elif Criteria =="D":
            Obj=np.log(np.linalg.det(ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon, dsphereum, catmassgram))) #we take log of determinant
        elif Criteria ==1:
            Obj=ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon, dsphereum, catmassgram)[0,0]
        elif Criteria ==2:
            Obj=ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon, dsphereum, catmassgram)[1,1]   
        elif Criteria ==3:
            Obj=ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon, dsphereum, catmassgram)[2,2]   
        else:
            e,v=np.linalg.eig(ExpectedCovFunction(PastCovariance, NewExp, TrueParams, epsilon, dsphereum, catmassgram))    
            Obj=np.max(e)
        return Obj
    
    #need to set a seed so I get the same results everytime so the work is reproducible
    def LatinGenerator(N_variables, N_sample, VarRanges):
        #np.random.seed(1) #This sets the seed for my random functions like the latin generator to make them give the same results everytime.
        sampling=pyDOE.lhs(N_variables, N_sample)
        Designs=np.zeros([N_sample,N_variables])
        for i in range (0, N_variables):
            Designs[:,i]=VarRanges[i,0]+sampling[:,i]*(VarRanges[i,1]-VarRanges[i,0])
        return Designs
    NumberofExpInScreen = 10000
    Screening=LatinGenerator(Number_variables,NumberofExpInScreen,VarRanges)
    #print Screening
    
    def FindBestGuess(Designs, PastCovariance, TrueParams, epsilon, Criteria, dsphereum, catmassgram):
        BestObjective=100
        BestDesign=0
        for i in range (0, len(Designs)): 
            CriteriaValue=ObjFunction(Designs[i,:], PastCovariance, TrueParams, epsilon, Criteria, dsphereum, catmassgram)
            if CriteriaValue < BestObjective:
                #print "We found a better guess"
                BestObjective = CriteriaValue
                BestDesign = i
        BestGuessDesign=Designs[BestDesign,:]
        return BestGuessDesign        
    BestDesignFromScreening=FindBestGuess(Screening, Cov, params, Disturbance, Criteria, dsphereum, catmassgram)
    #print BestDesignFromScreening
    
    def MBDoE(NewExp, PastCovariance, TrueParams, epsilon, Criteria, dsphereum, catmassgram):
        new_design=minimize(ObjFunction, NewExp, method = 'SLSQP', bounds = ([VarRanges[0,0],VarRanges[0,1]],[VarRanges[1,0],VarRanges[1,1]],[VarRanges[2,0],VarRanges[2,1]]), options = {'maxiter':10000, 'ftol':1e-20}, args = (PastCovariance, TrueParams, epsilon, Criteria,dsphereum, catmassgram,))
        return new_design    
    MinimisedMBDoE=MBDoE(BestDesignFromScreening, Cov, params, Disturbance, Criteria, dsphereum, catmassgram)
    NewDesign=MinimisedMBDoE.x
    Objvalue=MinimisedMBDoE.fun
    #print NewDesign
    
    return params, wt_residuals, chisq_ref, Cov, Corr, Cint, tVal, tRef, NewDesign

New_params, New_wt_residuals, New_chisq_ref, New_Cov, New_Corr, New_Cint, New_tVal, New_tRef, New_NewDesign=PE_MBDoE(ExpConditions, RealExpResultsFactorial1, ParameterGuess, Sigma, Variableranges, ObjCriteria, diametersphere, CatMass)

#Now I need to update the expconditions and result files
After2exp_ExpConditions=np.vstack((ExpConditions, New_NewDesign))

#I want to keep the same results from before, not generate new random ones every time. So just take the bottom row out
#print SimulatedExpResults
WarningResult=ModelplusError(After2exp_ExpConditions, TrueParameters, diametersphere, CatMass, Sigma)
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
def Repeat_PE_MBDoE(Exp, dsphereum, catmassgram, SimResults, thetaguess, stdev, VarRanges, Criteria, ListofPE, ListofX2, ListofX2ref, ListofCorr, ListofConfInt, ListofT, ListofTref):
    
    New_KP, New_X, New_X_ref, New_Covar, New_Correlation, New_CFint, New_tV, New_tR, New_MBDoEExp=PE_MBDoE(Exp, SimResults, thetaguess, stdev, VarRanges, Criteria, dsphereum, catmassgram)

    
    New_ExpConditions=np.vstack((Exp, New_MBDoEExp))
    
    #I want to keep the same results from before, not generate new random ones every time. So just take the bottom row out
    FakeResult=ModelplusError(New_ExpConditions, TrueParameters, diametersphere, CatMass, Sigma)
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
#Test=Repeat_PE_MBDoE(After2exp_ExpConditions, diametersphere, CatMass, After2exp_SimulatedExpResults, ParameterGuess, Sigma, Variableranges, ObjCriteria, After2exp_ListPE, After2exp_ListX2, After2exp_ListX2ref, After2exp_ListCorr, After2exp_ListConfInt, After2exp_ListT, After2exp_ListTref)

    
def NTimesMBDoE(NumberOfDesigns, ExpConditionsIn, dsphereum, catmassgram, SimulatedExpResultsIn, thetaguess, stdev, VarRanges, Criteria, ListofPEIn, ListofX2In, ListofX2refIn, ListofCorrIn, ListofConfIntIn, ListofTIn, ListofTrefIn):
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
        MBDoE_ExpConditions, MBDoE_SimulatedExpResults, ListPE, ListX2, ListX2ref, ListCorr, ListConfInt, ListT, ListTref=Repeat_PE_MBDoE(MBDoE_ExpConditions, dsphereum, catmassgram, MBDoE_SimulatedExpResults, thetaguess, stdev, VarRanges, Criteria, ListPE, ListX2, ListX2ref, ListCorr, ListConfInt, ListT, ListTref)

    return MBDoE_ExpConditions, MBDoE_SimulatedExpResults, ListPE, ListX2, ListX2ref, ListCorr, ListConfInt, ListT, ListTref

FinalValues=NTimesMBDoE(NumberofMBDoEExp, After2exp_ExpConditions, diametersphere, CatMass, After2exp_SimulatedExpResults, ParameterGuess, Sigma, Variableranges, ObjCriteria, After2exp_ListPE, After2exp_ListX2, After2exp_ListX2ref, After2exp_ListCorr, After2exp_ListConfInt, After2exp_ListT, After2exp_ListTref)

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
wswrite['D'+str(1)]="dsphere" 
wswrite['E'+str(1)]="cat mass"
wswrite['F'+str(1)]="BAC" 
wswrite['G'+str(1)]="EBC"  
wswrite['H'+str(1)]="KP1" 
wswrite['I'+str(1)]="KP2"
wswrite['J'+str(1)]="KP3"
wswrite['K'+str(1)]="KP4" 
wswrite['L'+str(1)]="X2" 
wswrite['M'+str(1)]="X2ref" 
wswrite['N'+str(1)]="KP1 tvalue" 
wswrite['O'+str(1)]="KP2 tvalue"
wswrite['P'+str(1)]="KP3 tvalue" 
wswrite['Q'+str(1)]="KP4 tvalue" 
wswrite['R'+str(1)]="tref"
wswrite['S'+str(1)]="KP1 CInt" 
wswrite['T'+str(1)]="KP2 CInt"
wswrite['U'+str(1)]="KP3 CInt" 
wswrite['V'+str(1)]="KP3 CInt" 

for i in range (0, len(FinalExpList)-1):
    wswrite['A'+str(i+2)]=FinalExpList[i,0] #using the excel numbering system    
    wswrite['B'+str(i+2)]=FinalExpList[i,1] #using the excel numbering system 
    wswrite['C'+str(i+2)]=FinalExpList[i,2] #using the excel numbering system 
    wswrite['D'+str(i+2)]=diametersphere #using the excel numbering system 
    wswrite['E'+str(i+2)]=CatMass #using the excel numbering system 
    wswrite['F'+str(i+2)]=FinalSimResults[i,0] #using the excel numbering system 
    wswrite['G'+str(i+2)]=FinalSimResults[i,1] #using the excel numbering system

for i in range (0, len(FinalX2)):
    wswrite['H'+str(i+9)]=FinalPE[i,0] #using the excel numbering system    
    wswrite['I'+str(i+9)]=FinalPE[i,1] #using the excel numbering system 
    wswrite['J'+str(i+9)]=FinalPE[i,2] #using the excel numbering system 
    wswrite['K'+str(i+9)]=FinalPE[i,3] #using the excel numbering system 
    #wswrite['L'+str(i+9)]=FinalX2[i] #using the excel numbering system 
    #wswrite['M'+str(i+9)]=FinalX2ref[i] #using the excel numbering system 
    wswrite['N'+str(i+9)]=Finaltvalues[i,0] #using the excel numbering system 
    wswrite['O'+str(i+9)]=Finaltvalues[i,1] #using the excel numbering system
    wswrite['P'+str(i+9)]=Finaltvalues[i,2] #using the excel numbering system
    wswrite['Q'+str(i+9)]=Finaltvalues[i,3] #using the excel numbering system
    #wswrite['R'+str(i+9)]=Finaltref[i] #using the excel numbering system
    wswrite['S'+str(i+9)]=FinalConfInt[i,0] #using the excel numbering system 
    wswrite['T'+str(i+9)]=FinalConfInt[i,1] #using the excel numbering system
    wswrite['U'+str(i+9)]=FinalConfInt[i,2] #using the excel numbering system
    wswrite['V'+str(i+9)]=FinalConfInt[i,3] #using the excel numbering system

#Saving File
wbwrite.save("Testing.xlsx")# overwrites without warning. So be careful

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