import math
import numpy as np
import random



__all__ = ["SOM"]

class SOM():
    """
    Kohonen SOM realisation. 
    """
    def __init__(self,N_nodes):
        """
        N_nodes - int or tuple  
        """       
        if N_nodes.__class__==int:
            M_nodes=1
        else:
            if N_nodes.__class__==tuple:
                M_nodes=N_nodes[1]
                N_nodes=N_nodes[0]
            
        self.N_nodes = N_nodes
        self.M_nodes = M_nodes
        self.N=N_nodes*M_nodes
        
        
        self.w_key=True


        
 


    def _get_range(self,x,jx,jw):
        s=((x[jx,:]-self.w[jw,:])**2).sum()
        return math.sqrt(s)

    def _get_min_index(self,x,ix,p=None):

        N=self.N
        
        result=0
        if p.__class__.__name__ == 'NoneType':
            rho_min=self._get_range(x,ix,0)
            for i in range(N-1):
                rho=self._get_range(x,ix,i+1)
                if rho_min>rho:
                    rho_min=rho
                    result=i+1
        else:
            i=0
            rho_min=self._get_range(x,ix,0)
            while p[i]<self.Pmin:
                i=i+1
                if i==N:
                    return -1
                rho_min=self._get_range(x,ix,i)
                result=i

            while i < (N-1):
                i=i+1
                rho=self._get_range(x,ix,i)
                if rho_min>rho and p[i]>self.Pmin:
                    rho_min=rho
                    result=i
                    
        return result

    def _correct(self,x, i, j):
        self.w[j,:]=self.a[j]*(x[i,:]-self.w[j,:])+self.w[j,:]
        self.a[j]=self.a[j]/(self.a[j]+1)


    def learn(self, x ,  N_iter_max=1000, lamda_start=1,lamda_end=0.0001 ,koef_min=0.1, koef_max=0.7, rn_max=100, _print=False, type_learn=1):
        """
        learn SOM
        x - data to learn [ array ]
        N_iter_max - maximum iteraction [ int ]
        lamda_start - start factor of learn [ float ]
        lamda_end - end factor of learn [ float ]
        
        koef_min -  w initilizate as random (koef_min - koef_max) [ float ]
        koef_max -  w initilizate as random (koef_min - koef_max) [ float ]
        
        rn_max - factor func distances between nodes [ float ]
        type_lern - 1 - simple learn, 2 - learn all nodes, 3 - uses and chenges istances between nodes [ 1 2 3 ]
        _print=False - True - print information
        """            

            
        self.R=len(x[0])
        self.N_iter_max=N_iter_max
        self.koef=koef_max
        self.koef0=koef_min
        
        self.lamda_start=lamda_start
        self.lamda_end=lamda_end
        

            
         
        self.a=np.zeros(self.N,dtype=float)+lamda_start
        self.lamda_end=lamda_end
   
        if self.w_key:
            self.w=np.random.random( (self.N,self.R) )*koef_max + koef_min
            self.w_key=False
        
        if _print:
            print("Коэфициенты до обучения:")
            print(self.w)
            print("Максимальное число итераций:" + str(N_iter_max))
        
        
        if type_learn==1:
            n=0
            M=len(x)
            lis=[]
            for i in range(M):
                lis.append(i)
            while self.a.max()>self.lamda_end and n<N_iter_max:
                random.shuffle(lis)
                for i in lis:
                    j=self._get_min_index(x,i)
                    self._correct(x,i,j)
                n+=1


        if type_learn==2:
            self.Pmin=( self.N-1)/ self.N
            p=np.zeros(self.N, dtype=float)+self.Pmin

 
            M=len(x)
            lis=[]
            for i in range(M):
                lis.append(i)
                
            n_winer=np.zeros(self.N_nodes)
            n=0

            while self.a.max()>self.lamda_end and n<N_iter_max:

                random.shuffle(lis)
                for i in lis:
                    j=self._get_min_index(x,i,p)
                    if j==-1:
                        p=p+1/self.N
                        continue
                    n_winer[j]+=1
                    p=p+1/self.N
                    p[j]=p[j]-self.Pmin-1/self.N
                    

                    self._correct(x,i,j)
                
 
                n+=1
                
            if _print:
                print(n)
                print(n_winer)


        if type_learn==3:
            self.rn_max=rn_max 
            self._placed_node()
            M=len(x)
            lis=[]
            for i in range(M):
                lis.append(i)

            n=0

            while n<N_iter_max:
                    
                random.shuffle(lis)
                for j in lis:
                    i_win=self._get_min_index(x,j)

                    rn=self._get_rn(i_win) #i,rho


                    
                    a1=self._ak(n)*(x[j]-self.w)
                    a2=np.exp(-rn/rn.max()*self._lk(n)).reshape((self.N,1))

                    self.w=self.w+a2*a1
                    
                n+=1
        if _print:
            print("Обученные коэфициенты:")
            print(self.w)

    def _get_rn(self,i_win):
        rho_node=((self.x_node-self.x_node[i_win])**2).sum(axis=1)
        #rho_node.sort()
        return rho_node

  
    def _ak(self,k):
        return self.lamda_start*((self.lamda_end/self.lamda_start)**(k/self.N_iter_max))

    def _lk(self,k):
        return ((self.rn_max)**(k/self.N_iter_max))

 



    def separate(self,x):
        """
        separate x between nodes.
        retuen array of nodes id
        """       
        M=len(x)
        result=np.zeros(M, dtype='int')
        for i in range(M):
            j=self._get_min_index(x,i)
            result[i]=j
        
        return result

    def _placed_node(self):
        r=0.5
        xy=np.zeros( (self.N_nodes*self.M_nodes,2 ))
        k=-1
        for i in range(self.N_nodes):
            for j in range(self.M_nodes):
                k += 1
                xy[k][0]=(j+1)*r
                xy[k][1]=(i+1)*r
        self.x_node=xy


                

