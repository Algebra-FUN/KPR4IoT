import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class IOT():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.neighbour=[]
        self.choose = None
        self.success = False

class RB():
    def __init__(self,rank):
        self.rank = rank
        self.num = 0

def setIOT(lambada,R):
    num = np.random.poisson(lambada*R**2,1)[0]
    x = np.random.rand(num)*R
    y = np.random.rand(num)*R
    iots = []
    for i in range(num):
        iots.append(IOT(x[i],y[i]))
    return iots

def setRB(num):
    rbs = []
    for i in range(num):
        rbs.append(RB(i))
    return rbs

def build_connection(iots,rc):
    for i in range(len(iots)):
        for j in range(len(iots)):
            if not i == j:
                dist = ((iots[i].x - iots[j].x)**2 + (iots[i].y - iots[j].y)**2)**0.5
                if dist <= rc:
                    iots[i].neighbour.append(iots[j])

def rand_choose(iots,rbs,prob):
    n = len(rbs)
    for rb in rbs:
        rb.num = 0
    for i in range(len(iots)):
        p = np.random.rand()
        if p <= prob:
            rank = np.random.randint(n) 
            rbs[rank].num+=1
            iots[i].choose = rbs[rank]
        else:
            iots[i].choose = None
            iots[i].success = False
    
    for i in range(len(iots)):
        if not iots[i].choose == None:
            if iots[i].choose.num == 1:
                iots[i].success = True
    
def foundvalidneighbour(iot):
    validneighbour = []
    for nei in iot.neighbour:
        if nei.choose:
            validneighbour.append(nei)
    return validneighbour

def rand_choose_rank(iot,rbs):
    p_no = np.random.rand()
    rank = len(rbs)-1
    if p_no<12/15:
        rank -= 1
    if p_no<9/15:
        rank-=1
    if p_no<6/15:
        rank-=1
    if p_no<3/15:
        rank-=1
    rbs[rank].num+=1
    iot.choose = rbs[rank]

def rule_choose(iots,rbs,prob):
    for rb in rbs:
        rb.num = 0
    for i in range(len(iots)):
        p = np.random.rand()
        if p <= prob:
            validneighbour = foundvalidneighbour(iots[i])
        
            if len(validneighbour) == 0:
                rand_choose_rank(iots[i],rbs)
                continue   
        
            failureIOT = []
            all_success_flag = True
            for iot in range(len(validneighbour)):
                if not validneighbour[iot].success:
                    all_success_flag = False
                    failureIOT.append(validneighbour[iot])

            if all_success_flag:
                rank = 0
                for iot in range(len(validneighbour)):
                    if validneighbour[iot].choose.rank >= rank:
                        rank = validneighbour[iot].choose.rank
                        # print(f"{validneighbour[iot].choose.rank} >= {rank}")
                chooserank = np.random.randint(rank, len(rbs), 1)[0]
            
            else:
                rank = len(rbs)-1
                for iot in range(len(failureIOT)):
                    if failureIOT[iot].choose.rank <= rank:
                        rank = failureIOT[iot].choose.rank   
                        # print(f"{failureIOT[iot].choose.rank} <= {rank}")
                chooserank = np.random.randint(0,rank+1, 1)[0]
                
            # print()
            # print(f"{i} choose {chooserank}")
            rbs[chooserank].num+=1
            iots[i].choose = rbs[chooserank]
        
        else:
            iots[i].choose = None
            iots[i].success = False     
    
    for i in range(len(iots)):
        if not iots[i].choose == None:
            if iots[i].choose.num == 1:
                iots[i].success = True

def simulation(p,num_of_rb,lambada,R,allrc,rounds,userule):
    results = []
    if userule:
        for rc in tqdm(np.linspace(1,allrc,50)):
            oneresults = 0
            for k in range(5):
                allservice = []
                iots = setIOT(lambada,R)
                build_connection(iots,rc)
                rbs = setRB(num_of_rb)
                rand_choose(iots,rbs,p)
                
                for t in range(rounds):
                    service = 0
                    rule_choose(iots,rbs,p)
                    for i in range(len(rbs)):
                        if rbs[i].num == 1:
                            service+=1
                    allservice.append(service)
                oneresults+=np.sum(np.array(allservice))/(rounds*num_of_rb)
            results.append(oneresults/5)
    else:
        for rc in tqdm(np.linspace(1,allrc,50)):
            oneresults = 0
            for k in range(5):
                allservice = []
                iots = setIOT(lambada,R)
                # build_connection(iots,rc)
                rbs = setRB(num_of_rb)
                rand_choose(iots,rbs,p)

                for t in range(rounds):
                    service = 0
                    rand_choose(iots,rbs,p)
                    for i in range(len(rbs)):
                        if rbs[i].num == 1:
                            service+=1
                    allservice.append(service)
                oneresults+=np.sum(np.array(allservice))/(rounds*num_of_rb)
            results.append(oneresults/5)

    return results
    

# allrc = 10
# x = np.linspace(1,allrc,50)

# results_p001_l2_5_random = simulation(p=0.01,num_of_rb=5,lambada=2.5,R=20,allrc=allrc,rounds=1000,userule=False)
# results_p001_l5_random= simulation(p=0.01,num_of_rb=5,lambada=5,R=20,allrc=allrc,rounds=1000,userule=False)
# results_p005_l2_5_random= simulation(p=0.05,num_of_rb=5,lambada=2.5,R=20,allrc=allrc,rounds=1000,userule=False)
# results_p005_l5_random= simulation(p=0.05,num_of_rb=5,lambada=5,R=20,allrc=allrc,rounds=1000,userule=False)

# results_p001_l2_5_rank = simulation(p=0.01,num_of_rb=5,lambada=2.5,R=20,allrc=allrc,rounds=1000,userule=True)
# results_p001_l5_rank = simulation(p=0.01,num_of_rb=5,lambada=5,R=20,allrc=allrc,rounds=1000,userule=True)
# results_p005_l2_5_rank = simulation(p=0.05,num_of_rb=5,lambada=2.5,R=20,allrc=allrc,rounds=1000,userule=True)
# results_p005_l5_rank = simulation(p=0.05,num_of_rb=5,lambada=5,R=20,allrc=allrc,rounds=1000,userule=True)




# plt.figure()
# plt.xlabel(r"Communication range $r_{c}$(m)")
# plt.ylabel("Service rate (%)")
# plt.plot(x, np.mean(results_p001_l2_5_random)+np.zeros(50), color="#0000FF", label=r"Baseline With $\lambda$=2.5")
# plt.plot(x, np.mean(results_p001_l5_random)+np.zeros(50), color="#0000FF",linestyle='--', label=r"Baseline With $\lambda = 5$")
# plt.plot(x, results_p001_l2_5_rank, color="#FF0000", label=r"Learning With $\lambda = 2.5$")
# plt.plot(x, results_p001_l5_rank, color="#FF0000",linestyle='--', label=r"Learning With $\lambda = 5$")
# plt.title("p = 0.01")
# plt.legend()
# plt.savefig("1.png")
# plt.close()



# plt.figure()
# plt.xlabel(r"Communication range $r_{c}$(m)")
# plt.ylabel("Service rate (%)")
# plt.plot(x, np.mean(results_p005_l2_5_random)+np.zeros(50), color="#0000FF", label=r"Baseline With $\lambda = 2.5$")
# plt.plot(x, np.mean(results_p005_l5_random)+np.zeros(50), color="#0000FF",linestyle='--', label=r"Baseline With $\lambda = 5$")
# plt.plot(x, results_p005_l2_5_rank, color="#FF0000", label=r"Learning With $\lambda= 2.5$")
# plt.plot(x, results_p005_l5_rank, color="#FF0000",linestyle='--', label=r"Learning With $\lambda = 5$")
# plt.title("p = 0.05")
# plt.legend()
# plt.savefig("2.png")
# plt.close()




