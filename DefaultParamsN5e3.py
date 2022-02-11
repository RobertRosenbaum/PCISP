import numpy as np
from numpy.linalg import inv
from time import time as tm


# Number of neurons in each population
Ne=4000
Ni=1000
N=Ne+Ni

Ne1=(int)(Ne/2)

# Recurrent net connection probabilities
P=np.array([[0.1, 0.1],
            [0.1, 0.1]])

# Mean connection strengths between each cell type pair
Jm=10*np.array([[50, -350],
                [225, -500]])/np.sqrt(N)

# Time (in ms) for sim
T=1000
Tburn=200
dt=.1
Nt=round(T/dt)
Nburn=round(Tburn/dt)
time = np.arange(dt, T+dt, dt)
dtRecord=dt#2

# Neuron parameters
taum=15
EL=-72
Vth=0
Vre=-73
DeltaT=2
VT=-55
Vlb=-80

# Synaptic time constants
taue=6
taui=4

# Plasticity params
etae=4000/np.sqrt(N)   # Learning rates
etai=.5*etae
tauSTDP=200 # eligibility trace timescales
r0e=4/1000  # Target rates
r0i=8/1000

# Connection matrices
start1 = tm()
Jee=Jm[0,0]*np.array(np.random.binomial(1, P[0,0], size=(Ne, Ne)), order = 'F')
Jei0=Jm[0,1]*np.array(np.random.binomial(1, P[0,1], size=(Ne, Ni)), order = 'F')
Jie=Jm[1,0]*np.array(np.random.binomial(1, P[1,0], size=(Ni, Ne)), order = 'F')
Jii0=Jm[1,1]*np.array(np.random.binomial(1, P[1,1], size=(Ni, Ni)), order = 'F')
stop1 = tm()
print("Time to generate connections: %.2f sec." %(stop1-start1))

# Jee[:Ne1,:Ne1]=Jee[:Ne1,:Ne1]*.8
# Jee[Ne1:,Ne1:]=Jee[Ne1:,Ne1:]*.8
# Jee[:Ne1,Ne1:]=Jee[:Ne1,Ne1:]*1.2
# Jee[Ne1:,:Ne1]=Jee[Ne1:,:Ne1]*1.2


# Maximum number of spikes for all neurons
# in simulation. Make it 50Hz across all neurons
# If there are more spikes, the simulation will
# terminate
maxns=np.int(np.ceil(.02*Ne*T))

# Number of matched and mismatched trials
numtrials=110

# Plasticity on or off
eiPlast=np.ones(numtrials)
iiPlast=np.ones(numtrials)

#eiPlast[:5]=0
#iiPlast[:5]=0

# Baseline external input
Xe0=0.6*np.sqrt(N)
Xi0=0.4*np.sqrt(N)

# From how many neurons to record all time steps
numerecord=5#Ne
numirecord=0#2

# From which trials to record all time steps
trialrecord=[*range(20)]+[*range(numtrials-20,numtrials)]
numtrialrec=len(trialrecord)
