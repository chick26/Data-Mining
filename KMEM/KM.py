import numpy as np
import time

class K_means(object):

    #initialization
    def __init__(self,k,data):
        self.k=k
        self.data=data
        self.label=np.zeros(len(data),dtype=np.int)
        self.SSE=np.zeros(len(data))
        #Random ini_centroids
        self.centroids=np.min(data)+(np.max(data)-np.min(data))*np.random.rand(k)
        
    #distance computation
    def Euclidean_distance(self,p1,p2):
        return np.sqrt(np.sum(p1-p2)**2)

    def Iteration(self):
        data=self.data
        SSE=np.zeros(len(data))
        while True:
            temp=self.centroids
            for var in range(len(data)):
                min_distance,min_index=np.inf,-1
                for cent in range(self.k):
                    distance=self.Euclidean_distance(self.centroids[cent],data[var])
                    if distance<min_distance:
                        min_distance,min_index=distance,cent
                        self.label[var]=min_index
                SSE[var]=self.Euclidean_distance(data[var],self.centroids[self.label[var]])**2
            #Update centroid
            for cent in range(self.k):
                self.centroids[cent]=np.mean(data[self.label==cent],axis=0)
            #
            if (temp==self.centroids).all():
                break
        return np.sum(SSE)
    
    def Find_bestSSE(self):
        min_SSE=np.inf
        for i in range(10):
            SSE=self.Iteration()
            if SSE < min_SSE:
                min_SSE=SSE
                best_centroids=self.centroids
                best_label=self.label
        best_cluster=[[]for n in range(max(self.label)+1)]

        #assignment of data
        for var in range(max(self.label)+1):
            best_cluster[var]=self.data[self.label==var]
        return min_SSE,best_centroids,best_label,best_cluster

class EM(object):

    #initialization
    def __init__(self,k,data):
        self.k=k
        self.data=data
        self.label=np.zeros(len(data),dtype=np.int)
        #random ini_mu & ini_sigma & ini_ratio
        self.mu=np.random.random(k)
        self.sigma_squre=np.random.random(k)
        self.ratio=np.array([1/k for n in range(k)])
        self.N=len(data)

    #Normal distribution
    def Normal(self,x,mu,sigma):
        return np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    #E step
    def Expectation(self):
        Expectations = np.zeros((self.N, self.k))
        for i in range(self.N):
            Down = 0
            for j in range(self.k):
                Down = Down + self.Normal(self.data[i],self.mu[j],np.sqrt(self.sigma_squre[j]))*self.ratio[j]
            for j in range(self.k):
                Up = self.Normal(self.data[i],self.mu[j],np.sqrt(self.sigma_squre[j]))*self.ratio[j]
                Expectations[i, j] = Up / Down
        return Expectations

    #M step
    def Maximum(self,Expectations):
        mu=np.random.random(self.k)
        for j in range(self.k):
            Up_mu,Down_mu,Up_sigma,Down_sigma=0,0,0,0
            #Updata mu,ratio,sigma
            for i in range(self.N):
                Up_mu += Expectations[i, j] * self.data[i]
                Down_mu += Expectations[i, j]
            self.mu[j] = Up_mu/Down_mu
            mu[j]=Up_mu/Down_mu
            self.ratio[j]=Down_mu/self.N
            for i in range(self.N):
                Up_sigma += Expectations[i, j] * (self.data[i]-self.mu[j])**2
                Down_sigma += Expectations[i, j]
            self.sigma_squre[j]=Up_sigma/Down_sigma
        return mu
    
    def Iteration(self):
        iter_num=0
        best_mu=np.random.random(self.k)*np.inf
        while True:
            iter_num+=1
            Expectations=self.Expectation()
            old_mu=self.Maximum(Expectations)
            #Use mu as convergence criteria
            if np.sum(abs(best_mu-old_mu))<1e-8:
                #print( "iteration : ", iter_num ) 
                cluster=[[]for n in range(np.size(Expectations,1))]    
                #predict
                for i in range(len(self.data)):
                    self.label[i]=np.argmax(Expectations[i])
                #assignment of data
                for var in range(np.size(Expectations,1)):
                    cluster[var]=self.data[self.label==var]
                break
            best_mu=old_mu
        return cluster


#Data Generate
def data_init(mu_1,mu_2,sigma_1,sigma_2,n1,n2):
    #np.random.seed(1)
    cluster_1=np.random.normal(mu_1, sigma_1, n1)
    cluster_2=np.random.normal(mu_2, sigma_2, n2)
    cluster=np.append(cluster_1,cluster_2)
    np.savetxt('TestData',cluster)
    np.savetxt('CheckCluster1',cluster_1)
    np.savetxt('CheckCluster2',cluster_2)
    return cluster_1,cluster_2,np.append(cluster_1,cluster_2)

#Accuracy
def accuracy(cluster,cluster_1,cluster_2):
    sum=0
    for i in range(2):
        sum+=max(len(np.intersect1d(cluster[i].flatten(),cluster_1)),len(np.intersect1d(cluster[i].flatten(),cluster_2)))/len(cluster_1)
    return sum/2

if __name__ == "__main__":
    #(cluster_1,cluster_2,cluster)=data_init(10,1,2,2,1000,1000)
    start=time.time()
    k=2

    #new dataset
    
    cluster=np.loadtxt('TestData')
    cluster_1=np.loadtxt('CheckCluster1')
    cluster_2=np.loadtxt('CheckCluster2')
    

    #input data array
    '''
    arr = input("") #1^2^3^5^7
    num = [float(n) for n in arr.split()]
    cluster=np.array(num)
    '''

    #input data file
    '''
    fliename=input("")
    cluster=np.loadtxt(str(fliename))
    '''
    #K_means
    KM=K_means(k,cluster)
    (min_SSE,best_centroids,best_label,best_cluster)=KM.Find_bestSSE()
    end=time.time()

    #calculate mean var
    print("K-means Test")
    for i in range(k):
        print("Mean,StdVar:\t",np.mean(best_cluster[i]),np.std(best_cluster[i],ddof=1))
        np.savetxt('cluster'+str(i),best_cluster[i])
    print("Time:\t\t",end-start)
    #print("Accuracy:\t",accuracy(np.array(best_cluster),np.array(cluster_1),np.array(cluster_2)))
    
    #EM
    '''
    start=time.time()
    EM=EM(k,cluster)
    cluster=EM.Iteration()
    end=time.time()
    print("EM Test")
    for i in range(k):
        print("Mean,StdVar:\t",np.mean(cluster[i]),np.std(cluster[i],ddof=1))
    print("Time:\t\t",end-start)
    #print("Accuracy:\t",accuracy(np.array(cluster),np.array(cluster_1),np.array(cluster_2)))
'''