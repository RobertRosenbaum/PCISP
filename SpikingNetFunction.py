import numpy as np
from time import time as tm
import math


###### Spiking net function
def MultiTrialSpikingNet0(Jee,Jei,Jie,Jii,Xe,Xi,taum,EL,Vth,Vre,DeltaT,VT,Vlb,taue,taui,r0e,r0i,tauSTDP,etae,etai,eiPlast,iiPlast,trialrecord,nerecord,nirecord,Nt,dt,Nburn,maxns,dtRecord):
  startsims = tm()
  Ne=np.shape(Jee)[0]
  Ni=np.shape(Jii)[0]
  N=Ne+Ni
  numtrials=np.shape(Xe)[1]
  T=Nt*dt
  Tburn=Nburn*dt

  nBinsRecord=round(dtRecord/dt)
  #timeRecord=np.arange(dtRecord, T+dtRecord, dtRecord)
  NtRec=int(np.ceil(Nt/nBinsRecord))
  numtrialrec=len(trialrecord)
  

  # Random initial voltages
  V0=np.random.uniform(0,1, size = N)*(VT-Vre)+Vre;


  # Integer division function
  IntDivide = lambda n,k: int((math.floor(n-1)/k))

  # Initialize arrays for storing time-averaged data for every trial and neuron
  MeanIee=np.zeros((numtrials,Ne), order = 'F')
  MeanIei=np.zeros((numtrials,Ne), order = 'F')
  MeanIex=np.zeros((numtrials,Ne), order = 'F')
  MeanIii=np.zeros((numtrials,Ni), order = 'F')
  MeanIie=np.zeros((numtrials,Ni), order = 'F')
  MeanIix=np.zeros((numtrials,Ni), order = 'F')
  AlleRates=np.zeros((numtrials,Ne), order = 'F')
  AlliRates=np.zeros((numtrials,Ni), order = 'F')

  # Initialize arrays for storing time-dependent data for some trials and neurons
  IeeRec=np.zeros((numtrialrec,nerecord,NtRec))
  IeiRec=np.zeros((numtrialrec,nerecord,NtRec))
  IexRec=np.zeros((numtrialrec,nerecord,NtRec))
  VeRec=np.zeros((numtrialrec,nerecord,NtRec))
  SeRec=np.zeros((numtrialrec,2,maxns))

  IieRec=np.zeros((numtrialrec,nirecord,NtRec))
  IiiRec=np.zeros((numtrialrec,nirecord,NtRec))
  IixRec=np.zeros((numtrialrec,nirecord,NtRec))
  ViRec=np.zeros((numtrialrec,nirecord,NtRec))
  SiRec=np.zeros((numtrialrec,2,maxns))


  # Initialize variables that are continuous across trials
  Ve=V0[:Ne]
  Vi=V0[Ne:]
  xe=np.zeros(Ne)
  xi=np.zeros(Ni)
  Iee=np.zeros(Ne)
  Iei=np.zeros(Ne)
  Iex=np.zeros(Ne)
  Iie=np.zeros(Ni)
  Iii=np.zeros(Ni)
  Iix=np.zeros(Ni)

  for ijk in range(numtrials):

    print(ijk,numtrials,end =" ")

    # Is this trial recorded?
    isRec=(ijk in trialrecord)
    
    # If so, store index
    if isRec:
      RecIndex=trialrecord.index(ijk)

    # Initialize variables
    IeeAvg=np.zeros(Ne)
    IeiAvg=np.zeros(Ne)
    IieAvg=np.zeros(Ni)
    IiiAvg=np.zeros(Ni)
    IexAvg=np.zeros(Ne)
    IixAvg=np.zeros(Ni)
    nespike=0
    nispike=0
    TooManySpikes=0
    se=-1+np.zeros((2,maxns))
    si=-1+np.zeros((2,maxns))
    for i in range(Nt): 

      # External inputs
      Iex=Xe[:,ijk]
      Iix=Xi[:,ijk]

      # Euler update to V
      Ve=Ve+(dt/taum)*(Iee+Iei+Iex+(EL-Ve)+DeltaT*np.exp((Ve-VT)/DeltaT))
      Vi=Vi+(dt/taum)*(Iie+Iii+Iix+(EL-Vi)+DeltaT*np.exp((Vi-VT)/DeltaT))
      Ve=np.maximum(Ve,Vlb)
      Vi=np.maximum(Vi,Vlb)
      

      # Find which E neurons spiked      
      Ispike = np.nonzero(Ve>=Vth)[0]        
      if Ispike.any():
          # Store spike times and neuron indices
          if nespike+len(Ispike)<=maxns :
              se[0,nespike:nespike+len(Ispike)]=dt*i
              se[1,nespike:nespike+len(Ispike)]=Ispike
          else:
              TooManySpikes=1
              #break

          # Reset e mem pot.
          Ve[Ispike]=Vre

          # Update exc synaptic currents
          Iee=Iee+Jee[:,Ispike].sum(axis = 1)/taue
          Iie=Iie+Jie[:,Ispike].sum(axis = 1)/taue
      
          # Update cumulative number of e spikes
          nespike=nespike+len(Ispike)
              
          # If there is plasticity onto e neurons
          if(eiPlast[ijk]>.1):
              #Jei[Ispike,:]=Jei[Ispike,:]-np.tile(etae*np.transpose(xi),(len(Ispike),1))
              Jei[Ispike,:]=Jei[Ispike,:]-etae*xi*(Jei[Ispike,:]!=0)
              Jei[Ispike,:]=np.minimum(Jei[Ispike,:],0)

          # Update rate estimates for plasticity rules
          xe[Ispike]=xe[Ispike]+1/tauSTDP
          
      # Find which I neurons spiked      
      Ispike=np.nonzero(Vi>=Vth)[0] 
      if Ispike.any():
          # Store spike times and neuron indices
          if nispike+len(Ispike)<=maxns :
              si[0,nispike:nispike+len(Ispike)]=dt*i
              si[1,nispike:nispike+len(Ispike)]=Ispike
          else:
              TooManySpikes=1
              #break
          
          # Reset i mem pot.
          Vi[Ispike]=Vre
      
          # Update inh synaptic currents
          Iei=Iei+Jei[:,Ispike].sum(axis = 1)/taui
          Iii=Iii+Jii[:,Ispike].sum(axis = 1)/taui
      
          # Update cumulative number of i spikes
          nispike=nispike+len(Ispike)
              
          # If there is plasticity onto i neurons
          if(iiPlast[ijk]):
              #Jii[Ispike,:]=Jii[Ispike,:]-np.tile(etai*np.transpose(xi),(len(Ispike),1))            
              Jii[Ispike,:]=Jii[Ispike,:]-(etai*xi)*(Jii[Ispike,:]!=0)
              Jii[Ispike,:]=np.minimum(Jii[Ispike,:],0)

              #Jii[:,Ispike]=Jii[:,Ispike]-np.transpose(np.tile(etai*(xi-2*r0i),(len(Ispike),1)))            
              Jii[:,Ispike]=Jii[:,Ispike]-etai*(xi[:,np.newaxis]-2*r0i)*(Jii[:,Ispike]!=0)
              Jii[:,Ispike]=np.minimum(Jii[:,Ispike],0)
          if(eiPlast[ijk]):
              #Jei[:,Ispike]=Jei[:,Ispike]-np.transpose(np.tile(etae*(xe-2*r0e),(len(Ispike),1)))
              Jei[:,Ispike]=Jei[:,Ispike]-etae*(xe[:,np.newaxis]-2*r0e)*(Jei[:,Ispike]!=0)
              Jei[:,Ispike]=np.minimum(Jei[:,Ispike],0)

          # Update rate estimates for plasticity rules
          xi[Ispike]=xi[Ispike]+1/tauSTDP
      
      
      # Euler update to synaptic currents
      Iee=Iee-dt*Iee/taue
      Iei=Iei-dt*Iei/taui      
      Iie=Iie-dt*Iie/taue      
      Iii=Iii-dt*Iii/taui
      
      # Update time-dependent firing rates for plasticity
      xe=xe-dt*xe/tauSTDP
      xi=xi-dt*xi/tauSTDP 

      # Keep track of averages
      if i>Nburn:
        IeeAvg=IeeAvg+Iee/(Nt-Nburn)
        IeiAvg=IeiAvg+Iei/(Nt-Nburn)
        IexAvg=IexAvg+Iex/(Nt-Nburn)
        IieAvg=IieAvg+Iie/(Nt-Nburn)
        IiiAvg=IiiAvg+Iii/(Nt-Nburn)
        IixAvg=IixAvg+Iix/(Nt-Nburn)

      # If this trial is recorded, store recorded variables      
      if isRec:      
        ii=IntDivide(i,nBinsRecord)
        IeeRec[RecIndex,:,ii]+=Iee[:nerecord]
        IeiRec[RecIndex,:,ii]+=Iei[:nerecord]
        IexRec[RecIndex,:,ii]+=Iex[:nerecord]
        VeRec[RecIndex,:,ii]+=Ve[:nerecord]
        IieRec[RecIndex,:,ii]+=Iie[:nirecord]
        IiiRec[RecIndex,:,ii]+=Iii[:nirecord]
        IixRec[RecIndex,:,ii]+=Iix[:nirecord]
        ViRec[RecIndex,:,ii]+=Vi[:nirecord]
    # End of time loop

    # Store means for this trial
    MeanIee[ijk,:]=IeeAvg
    MeanIei[ijk,:]=IeiAvg
    MeanIii[ijk,:]=IiiAvg
    MeanIie[ijk,:]=IieAvg
    MeanIex[ijk,:]=IexAvg
    MeanIix[ijk,:]=IixAvg

    # Store rates for this trial
    AlleRates[ijk,:]=np.histogram(se[1,np.logical_and(se[1,:]>=0, se[0,:]>=Tburn)],bins = range(Ne+1))[0]/(T-Tburn)
    AlliRates[ijk,:]=np.histogram(si[1,np.logical_and(si[1,:]>=0, si[0,:]>=Tburn)],bins = range(Ni+1))[0]/(T-Tburn)  

    # Store spike trains for this trial if recorded
    if isRec:
      SeRec[RecIndex,:,:]=se.copy()
      SiRec[RecIndex,:,:]=si.copy()

    # Print progress
    print('%.0f'%(tm()-startsims),'%.1f'%(1000*AlleRates[ijk,:].mean()),'%.1f;'%(1000*AlliRates[ijk,:].mean()), end =" ")
    if np.mod(ijk,4)==0:
      print('')

  IeeRec=IeeRec/nBinsRecord
  IeiRec=IeiRec/nBinsRecord
  IexRec=IexRec/nBinsRecord
  VeRec=VeRec/nBinsRecord
  IieRec=IieRec/nBinsRecord
  IiiRec=IiiRec/nBinsRecord
  IixRec=IixRec/nBinsRecord
  ViRec=ViRec/nBinsRecord
  
  return MeanIee,MeanIei,MeanIie,MeanIii,AlleRates,AlliRates,IeeRec,IeiRec,IexRec,VeRec,SeRec,SiRec

######




