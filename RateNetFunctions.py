import numpy as np
from numpy.linalg import inv
from time import time as tm
import numba
from numba import jit
from scipy.linalg import null_space

@jit(nopython=True)
def MultiTrialRateSimJit1(Wee,Wei,Wie,Wii,Nt,dt,taue,taui,Xe,Xi,etaE,etaI,r0e,r0i,eiPlast,iiPlast,ge,gi,eth,ith,reInit,riInit):

  Me=np.shape(Wee)[0]
  Mi=np.shape(Wii)[0]
  numtrials=np.shape(Xe)[1]

  # Recorded variables. Let's record all trials, all neurons, all time for now.
  # If this gets too big, we can change it.
  IeeRec=np.zeros((numtrials,Nt,Me))
  IeiRec=np.zeros((numtrials,Nt,Me))
  IexRec=np.zeros((numtrials,Nt,Me))
  reRec=np.zeros((numtrials,Nt,Me))
  
  IieRec=np.zeros((numtrials,Nt,Mi))
  IiiRec=np.zeros((numtrials,Nt,Mi))
  IixRec=np.zeros((numtrials,Nt,Mi))
  riRec=np.zeros((numtrials,Nt,Mi))


    
  ## Initialize r's
  re=np.zeros(Me)
  re[:]=reInit+np.zeros(Me)
  ri=np.zeros(Mi)
  ri[:]=riInit+np.zeros(Mi)

  for k in range(numtrials):
    for i in range(Nt):

        # Update inputs
        Iee = Wee@re
        Iei = Wei@ri        
        Iie = Wie@re
        Iii = Wii@ri
        
        # Total input to each pop
        Ie=Iee+Iei+Xe[:,k]
        Ii=Iie+Iii+Xi[:,k]

        # Update rates
        re = re+dt*(1/taue)*(-re+ge*Ie*(Ie>eth))
        ri = ri+dt*(1/taui)*(-ri+gi*Ii*(Ii>ith))
    
        # ISP updating rules
        if (eiPlast[k]):
            Wei = np.minimum(0,Wei - dt*etaE*np.outer(re-r0e, ri))
        if (iiPlast[k]):
            Wii = np.minimum(0,Wii - dt*etaI*np.outer(ri-r0i, ri))
        
        # Record variables
        IeeRec[k,i,:]=Iee
        IeiRec[k,i,:]=Iei
        IexRec[k,i,:]=Xe[:,k] # This isn't necessary to save, but makes stuff easier
        reRec[k,i,:]=re
        IieRec[k,i,:]=Iie
        IiiRec[k,i,:]=Iii
        IixRec[k,i,:]=Xi[:,k]
        riRec[k,i,:]=ri        
    
    if(k==1 or np.mod(k,100)==0 and k>0):    
      print('    Trial',k,'out of',numtrials,'=',100*k/numtrials,'percent done')
  return  reRec,riRec,IeeRec,IeiRec,IexRec,IieRec,IiiRec,IixRec,Wei,Wii



@jit(nopython=True)
def MultiTrialRateSimJitBal(W,X,Me,Nt,dt,etaE,etaI,r0,eiPlast,iiPlast,recordW):
    M=np.shape(W)[0]
    Mi=M-Me
    numtrials=np.shape(X)[1]

    # Recorded variables. Let's record all trials, all neurons, all time for now.
    # If this gets too big, we can change it.
    rRec=np.zeros((numtrials,M))  
    Wrec=np.zeros((numtrials,M,M))

    ## Initialize r's
    r=np.zeros(M)
    for k in range(numtrials):
        r=-np.linalg.pinv(W)@X[:,k]
        for i in range(Nt):
            # ISP updating rules
            if (eiPlast[k]):
                W[:Me,Me:] = np.minimum(0,W[:Me,Me:] - dt*etaE*np.outer(r[:Me]-r0[:Me], r[Me:]))
            if (iiPlast[k]):
                W[Me:,Me:] = np.minimum(0,W[Me:,Me:] - dt*etaI*np.outer(r[Me:]-r0[Me:], r[Me:]))

        rRec[k,:]=r
        if recordW:
          Wrec[k,:,:]=W

        if(k==1 or np.mod(k,1000)==0 and k>0):    
          print('    Trial',k,'out of',numtrials,'=',100*k/numtrials,'percent done')


    Wrec[k,:,:]=W

    return  rRec,Wrec
  



#@jit(nopython=True)
def MultiTrialRateSimJitSemiBal1(W,X,Me,Nt,dt,etaE,etaI,r0,eiPlast,iiPlast,recordW):
    M=np.shape(W)[0]
    Mi=M-Me
    numtrials=np.shape(X)[1]
    whichmethod=np.zeros(numtrials)
    # Recorded variables. Let's record all trials, all neurons, all time for now.
    # If this gets too big, we can change it.
    rRec=np.zeros((numtrials,M))  
    Wrec=np.zeros((numtrials,M,M))
    ## Initialize r's
    r=np.zeros(M)
    for k in range(numtrials):
        doSemiBal=False
        try: 
            r=-np.linalg.inv(W)@X[:,k]
            if np.min(r)>0:
                whichmethod[k]=0 # Regular balance
            else:        
                whichmethod[k]=1 # Regular semi-balance
                doSemiBal=True
                S=np.nonzero(r>0)[0]
        except:    
            v0=null_space(W.T)
            proj0=X[:,k]@v0/np.linalg.norm(X[:,k])
            if np.max(np.abs(proj0))<1e-3:
                whichmethod[k]=2 # Singular balance
                r=-np.linalg.pinv(W)@X[:,k]
            else:
                whichmethod[k]=3 # Singular semi-balance
                doSemiBal=True
                r=(np.linalg.inv(.001*np.eye(M)-W)@X[:,k])
                S=np.nonzero(r>0)[0]
        if doSemiBal: # Look for a semi-bal solution
            r=np.zeros(M)
            XS=X[S,k]
            WSS=W[np.ix_(S, S)]
            r[S]=-np.linalg.pinv(WSS)@XS
            if np.min(r[S])<=0:        
                print('Error: Did not find semi-bal solution on trial',k)

#        for i in range(Nt):
            # ISP updating rules
        if (eiPlast[k]):
                W[:Me,Me:] = np.minimum(0,W[:Me,Me:] - dt*Nt*etaE*np.outer(r[:Me]-r0[:Me], r[Me:]))
        if (iiPlast[k]):
                W[Me:,Me:] = np.minimum(0,W[Me:,Me:] - dt*Nt*etaI*np.outer(r[Me:]-r0[Me:], r[Me:]))

        rRec[k,:]=r
        if recordW:
          Wrec[k,:,:]=W

        if(k==1 or np.mod(k,1000)==0 and k>0):    
          print('    Trial',k,'out of',numtrials,'=',100*k/numtrials,'percent done')


    Wrec[k,:,:]=W

    return  rRec,Wrec,whichmethod

    
@jit(nopython=True)
def MultiTrialRateSimJit2(W,X,Me,T,etaE,etaI,r0,eiPlast,iiPlast,g,recordW):
  M=np.shape(W)[0]
  Mi=M-Me
  numtrials=np.shape(X)[1]

  # Recorded variables. Let's record all trials, all neurons, all time for now.
  # If this gets too big, we can change it.
  rRec=np.zeros((numtrials,M))  
  Wrec=np.zeros((numtrials,M,M))

  ## Initialize r's
  r=np.zeros(M)

  for k in range(numtrials):
    A=np.eye(M)-g*W
    b=g*X[:,k]
    r=np.maximum(np.linalg.solve(A,b),0)
    

    # ISP updating rules
    if (eiPlast[k]):
        W[:Me,Me:] = np.minimum(0,W[:Me,Me:] - T*etaE*np.outer(r[:Me]-r0[:Me], r[Me:]))
    if (iiPlast[k]):
        W[Me:,Me:] = np.minimum(0,W[Me:,Me:] - T*etaI*np.outer(r[Me:]-r0[Me:], r[Me:]))
    
    rRec[k,:]=r
    if recordW:
      Wrec[k,:,:]=W
    
    if(k==1 or np.mod(k,1000)==0 and k>0):    
      print('    Trial',k,'out of',numtrials,'=',100*k/numtrials,'percent done')
  
  
  Wrec[k,:,:]=W

  return  rRec,Wrec



#@jit(nopython=True)
def MultiTrialRateSimJit3(W,X,Me,T,etaE,etaI,r0,eiPlast,iiPlast,g,recordW):
  M=np.shape(W)[0]
  Mi=M-Me
  numtrials=np.shape(X)[1]

  # Recorded variables. Let's record all trials, all neurons, all time for now.
  # If this gets too big, we can change it.
  rRec=np.zeros((numtrials,M))  
  Wrec=np.zeros((numtrials,M,M))

  ## Initialize r's
  r=np.zeros(M)

  for k in range(numtrials):
    A=np.eye(M)-g*W
    b=g*X[:,k]
    r=np.linalg.solve(A,b)
    
    if np.min(r)<=0:
        S=np.nonzero(r>0)[0]
        r=np.zeros(M)
        bS=b[S]
        ASS=A[np.ix_(S, S)]        
        r[S]=np.linalg.solve(ASS,bS)
        print('r<0  ',np.min(r))
    

    # ISP updating rules
    if (eiPlast[k]):
        W[:Me,Me:] = np.minimum(0,W[:Me,Me:] - T*etaE*np.outer(r[:Me]-r0[:Me], r[Me:]))
    if (iiPlast[k]):
        W[Me:,Me:] = np.minimum(0,W[Me:,Me:] - T*etaI*np.outer(r[Me:]-r0[Me:], r[Me:]))
    
    rRec[k,:]=r
    if recordW:
      Wrec[k,:,:]=W
    
    if(k==1 or np.mod(k,1000)==0 and k>0):    
      print('    Trial',k,'out of',numtrials,'=',100*k/numtrials,'percent done')
  
  
  Wrec[k,:,:]=W

  return  rRec,Wrec
  