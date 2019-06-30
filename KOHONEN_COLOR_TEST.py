import tkinter as TK
import time
import math  



__all__ = ["Canvas_sg"]


class Canvas_sg(TK.Canvas):
    """
    Class to plot colors  
    """
    def initialisation(self,X=500,Y=500):
        self.X=X
        self.Y=Y
        self.config(width = X, height = Y)
        self.curent_k=-1
    
    def delete_plot(self):
        try:
            for i in range(len(self.canvas_id)):
                self.delete(self.canvas_id[i])
        except:
            None
            
    def get_color(self,x):
        a="#%02x%02x%02x" % (int(x[0]*256),int(x[1]*256),int(x[2]*256))
        return a
          
    def start_plot(self,data,w_date):
        k=self.curent_k
        while True:
            if k!=self.curent_k:
                self.delete_plot()
                k=self.curent_k
                N=len(data[k-1])
                if N==0:
                    continue
                while True:
                    sqrn=int(math.sqrt(N))
                    if sqrn**2!=N:
                        a=sqrn+1
                        b=int(N/a)
                        if a*b!=N:
                            b+=1
                    else:
                        a=sqrn
                        b=a

                    if a<self.X and b<self.Y:
                        break
                    else:
                        N=int(0.9*N)

                hx=int(self.X/a)
                hy=int(self.Y/b)



                
                canvas_id={}
                s=0
                for i in range(a):
                    for j in range(b):
                        canvas_id[s]=self.create_line(hx*(i+0.5),self.Y-hy*j,hx*(i+0.5),self.Y-hy*(j+1),width=hx,fill=self.get_color(data[k-1][s]))
                        s+=1
                        if s==len(data[k-1]):
                            break
                    if s==len(data[k-1]):
                        break
                self.canvas_id=canvas_id
            time.sleep(0.5)
  



if __name__ == '__main__':
    import numpy as np
    import _thread
    import Kohonen_SOM 
    
    N=10 #nomber nodes on X
    M=10 #nomber nodes on Y
    N_samples=N*M*3 #total nomber of sampels
    
    X = np.random.random_sample( (N_samples, 3) )

    R = len(X[0])

    test_SOM = Kohonen_SOM.SOM( (N,M))

    test_SOM.learn(X, N_iter_max=100, type_learn=3)
    is_class = test_SOM.separate(X)
    data={}
    w_date={}
    for j in range(M*N):
        data[j]={}
        nj=0
        w_date[j]={}
        for s in range(R):
            w_date[j][s]=test_SOM.w[j][s]
        for i in range(len(X)):
            if is_class[i] == j:
                data[j][nj]={}
                for s in range(R):
                    data[j][nj][s] = X[i][s]
                nj+=1
        

    root = TK.Tk()

    Tframe=TK.Frame(root)
    Bframe=TK.Frame(root)
    slice_frame={}
    for i in range(M):
        slice_frame[i]=TK.Frame(Bframe)

    
    canv=Canvas_sg(Tframe,  bg = "white")

    canv.initialisation()
    
    def push(k):
        #print(k)
        canv.curent_k=k


    PUSK={}
    for i in range(N):
        PUSK[i]={}
        for j in range(M):
            k=i*(M)+j+1
            li=len(str(k))
            lend=len(str(M*N))
            add='  '
            plus=''
            for l in range(lend-li):
                plus=plus+add
            PUSK[i][j]=TK.Button(slice_frame[j],text=(str(k)+plus),bg=canv.get_color(w_date[k-1]),command=lambda k=k : push(k),bd=2)
 
    Tframe.pack(side="top")
    Bframe.pack(side="bottom")
    
    for i in range(M):
        slice_frame[i].pack(side="top")

    for i in range(N):
        for j in range(M):
            PUSK[i][j].pack(side="left")

    canv.pack(fill='both',expand=True)

    _thread.start_new_thread(canv.start_plot,(data,0))
    root.mainloop()









