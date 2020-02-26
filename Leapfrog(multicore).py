 # -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 20:12:46 2020

@author: jonathan Beaulieu-Emond
"""
#Importation des fonctions
import numpy as np
from numba import njit,prange,jit
from scipy.constants import pi
import matplotlib.pyplot as plt
import time
from matplotlib import cm
#import plotly.plotly as py
#from plotly.grid_objs import Grid, Column

data=np.loadtxt('data.txt')




############-Périodicité-###############
@njit
def periodic(f,N):
    
    f[1:N+1,0]=f[1:N+1,N]
    f[1:N+1,N+1]=f[1:N+1,1]
    f[0,1:N+1]= f[N,1:N+1]
    f[N+1,1:N+1]=f[1,1:N+1]
    return f
   
#########Vitesse#########
@njit
def vitesse(x,y,t,N,omega,eps,A) :

        ux1=A*np.cos(y+eps*np.sin(omega*t))
        
        uy1=A*np.sin(x+eps*np.cos(omega*t))
        
        ux=np.ones((N,N))*ux1
        
        uy=np.ones((N,N))*uy1
        
        return(ux,uy)
    


#########Advection Leapfrog###########
@njit
def advectif_leapfrog(c,N,D,dt,dx,ux,uy) :
    return -dt*(ux*(c[2:N+2,1:N+1] - c[0:N,1:N+1])/dx + uy*(c[1:N+1,2:N+2] - c[1:N+1,0:N])/dx )
                   
  
#########Diffusion Leapfrog##########
@njit
def diffusif_leapfrog(c,N,D,dt,dx):
    return D*dt/dx**2*((c[1:N+1,2:N+2] + c[2:N+2,1:N+1] - 4*c[1:N+1,1:N+1] + c[1:N+1,0:N] + c[0:N,1:N+1]))
    

##########-Principal-###############   
@njit
def main(s0,x0,y0,td) :
    
    #####Paramètre de maille###########
    N=101
    dt=1e-3
    dx=2*pi/(N-1)
   
    tmax=4
    NITER=np.int(tmax/dt)+1
    
   
    #####-Paramètre de diffusion-##########
    A=-np.sqrt(6)
    
    w=0.25#source
    omega=5#vitesse
   
    x0,y0=x0*(N-1)*dx,y0*(N-1)*dx
    #td=0.01
    tf=2.5
    eps=1  #?????????
 
    Pe=1000
    D=1/Pe
    #s0=2 #???????????6
    ######-Boucle d'iteration-##########
    c=np.zeros((N+2,N+2))
    cnp1=np.zeros((N+2,N+2))
    cnm1=np.zeros((N+2,N+2))
    point1=np.zeros((NITER))
    point2=np.zeros((NITER))
    point3=np.zeros((NITER))
    
    
    #######-Condition de courant -############
    """
     C=np.sqrt(vx**2+vy**2)*dt/(dx**2)
    erreur=1-C**2*(1-C**2)*(1-np.cos(k*dx))**2
    if erreur>1 : 
        erreur=True
        print('fuck')
    else : erreur =False
    """
     
    xtemp=np.arange(0,N)*dx
    x = np.transpose(np.ones((N,N))*xtemp)
    y = np.ones((N,N))*xtemp
    #ytemp=np.arange(0,N)*dx
    r2=np.zeros((N,N))
    
    for i in prange(0,N):
        for j in prange(0,N) :
            r2[i][j]=(xtemp[i]-x0)**2+(xtemp[j]-y0)**2
    Source_precalc=s0*np.exp(-r2/w**2)
    
    
    for iteration in range(0,NITER) :
        #print(np.max(c),iteration)
        t=iteration*dt
        
        ux,uy=vitesse(x,y,t,N,omega,eps,A)
        
        advection_leapfrog=advectif_leapfrog(c,N,D,dt,dx,ux,uy)
        
        diffusion_leapfrog=diffusif_leapfrog(c,N,D,dt,dx)
        
        #Méthode de leapfrog avec diffusion FTCS
        cnp1[1:N+1,1:N+1]=diffusion_leapfrog+advection_leapfrog+cnm1[1:N+1,1:N+1]
             
        if td<=t<=tf :
            cnp1[1:N+1,1:N+1]+=2*dt*Source_precalc
        
        
          
        cnm1=np.copy(c)
        
        c=np.copy(periodic(cnp1,N))
        
        point1[iteration]=c[np.int(0.75*(N-1))+1][np.int(25/100*(N-1))+1]
        point2[iteration]=c[np.int(50/100*(N-1))+1][np.int(50/100*(N-1))+1]
        point3[iteration]=c[np.int(25/100*(N-1))+1][np.int(75/100*(N-1))+1]
        
        if t==0.5 : c1=np.copy(c[1:N])
        if t==1 : c2=np.copy(c[1:N])
        if t==2 : c3=np.copy(c[1:N])
        if t==3 : c4=np.copy(c[1:N])
        
    return   c[1:N],point1,point2,point3,c1,c2,c3,c4
   

"""

#for i in range(0,)
t1=time.time()
matrix,p1,p2,p3,c1,c2,c3,c4=main()
matrix=np.matrix.transpose(matrix) ####!!!!!!!!!!!!!! Attention la matrice doit être transposer!!
matrix=np.flip(matrix,0)
print(time.time()-t1) 

plt.subplots(nrows=2, ncols=2,sharex=True,sharey=True)
plt.subplots_adjust(hspace=0.3)


#c1/=np.max(c1)
#c2/=np.max(c2)
#c3/=np.max(c3)
#c4/=np.max(c4)
c1=np.flip(np.transpose(c1),0)
c2=np.flip(np.transpose(c2),0)
c3=np.flip(np.transpose(c3),0)
c4=np.flip(np.transpose(c4),0)
plt.subplot(221)
plt.imshow(c1,extent=([0,1,0,1]))
plt.subplot(222)
plt.imshow(c2,extent=([0,1,0,1]))
plt.subplot(223)
plt.imshow(c3,extent=([0,1,0,1]))
plt.subplot(224)
plt.imshow(c4,extent=([0,1,0,1]))
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.show()





z=np.arange(0.1,1,0.05)
#plt.figure(dpi=800)

plt.contour(matrix/np.max(matrix),z)
plt.show()

x=np.arange(0,len(p1))/1000
plt.plot(data[:,0],data[:,1],label='dataset')
plt.plot(data[:,0],data[:,2],label='dataset')
plt.plot(data[:,0],data[:,3],label='dataset')

plt.plot(x,p1,label='p1')
plt.plot(x,p2,label='p2')
plt.plot(x,p3,label='p3')
plt.legend()
"""

def graph(s0,x0,y0,td) :   
    matrix,p11,p22,p33,c1,c2,c3,c4=main(s0,x0,y0,td)
    c1=np.flip(np.transpose(c1),0)
    c2=np.flip(np.transpose(c2),0)
    c3=np.flip(np.transpose(c3),0)
    c4=np.flip(np.transpose(c4),0)
    plt.subplot(221)
    plt.imshow(c1,extent=([0,1,0,1]))
    plt.subplot(222)
    plt.imshow(c2,extent=([0,1,0,1]))
    plt.subplot(223)
    plt.imshow(c3,extent=([0,1,0,1]))
    plt.subplot(224)
    plt.imshow(c4,extent=([0,1,0,1]))
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()
    
    x=np.arange(0,len(p11))/1000
    
    plt.plot(data[:,0],data[:,1],label='dataset')
    plt.plot(data[:,0],data[:,2],label='dataset')
    plt.plot(data[:,0],data[:,3],label='dataset')
   
    
    plt.plot(x,p11,label='x=25,y=75')
    plt.plot(x,p22,label='x=50,y=50')
    plt.plot(x,p33,label='x=75,y=25')
    plt.title('Concentration en fonction du temps')
    plt.xlabel('temps')
    plt.ylabel('Concentration')
    plt.legend()

@njit(parallel=True)
def boucle(xf,yf,tdf,s0f,chif,data,taille,iteration4real,x00,y00,td00,s00) :
        for iteration in prange(0,taille) :
            #print(iteration,iteration4real)
            
            s0test,x0test,y0test,tdtest=seed(x00,y00,td00,s00,iteration4real)
            matrix,p1,p2,p3,c1,c2,c3,c4=main(s0test,x0test,y0test,tdtest)
            
            mp1=np.max(p1)
            mp2=np.max(p2)
            mp3=np.max(p3)
            if mp1==0 :
                mp1=1
            if mp2==0 :
                mp2=1
            if mp3==0 :
                mp3=1
            chif[iteration]=np.sum(1/len(p1)*(p1/mp1-data[:,1]/np.max(data[:,1]))**2)+np.sum(1/len(p1)*(p2/mp2-data[:,2]/np.max(data[:,2]))**2)+np.sum(1/len(p1)*(p3/mp3-data[:,3]/np.max(data[:,3]))**2)
        
            xf[iteration]=x0test
            yf[iteration]=y0test
            tdf[iteration]=tdtest
            s0f[iteration]=s0test
       
        chi1=np.min(chif)
        popt=np.where(chif==chi1)[0]
        
        x00,y00,td00,s00=xf[popt][0],yf[popt][0],tdf[popt][0],s0f[popt][0]
        
        return chi1,x00,y00,td00,s00
    
    
def optimisation(data) :
    
    x00=0.7340
    y00=0.11
    td00= 0.54919
    
    s00=1
    taille=400
    chi2=10000
    for iteration4real in range(1,10000) :
        xf=np.zeros((taille))
        yf=np.zeros((taille))
        tdf=np.zeros((taille))
        s0f=np.zeros((taille))
        chif=np.zeros((taille))
        chi1,x00,y00,td00,s00=boucle(xf,yf,tdf,s0f,chif,data,taille,iteration4real,x00,y00,td00,s00)
        if chi1<chi2 :
            chi2=chi1
            x0f2,s0f2,y0f2,td0f2=x00,s00,y00,td00
            print( x0f2,s0f2,y0f2,td0f2,chi2)
            #graph(1,x0f2,y0f2,td0f2)
                
    return x0f2,s0f2,y0f2,td0f2,chi2
@njit
def seed(x0,y0,td,s0,iteration) :
         if iteration==0 :
             x0test=np.random.uniform(0,1)
             y0test=np.random.uniform(0,1)
             tdtest=np.random.uniform(0,0.5)
             #s0test=np.random.uniform(0,10)
        
         else :
             #s0max=s0/iteration
             x0max=x0/iteration/4
             y0max=y0/iteration/4
             tdmax=td/iteration/4
             
             x0test=np.random.normal(x0,x0max)
             y0test=np.random.normal(y0,y0max)
             tdtest=np.random.normal(td,tdmax)
             #s0test=np.random.normal(s0,s0max)
        
         return 1,x0test,y0test,tdtest

data=np.loadtxt('data.txt')
#x00,s00,y00,td00,chi1=optimisation(data)
#print(x00,s00,y00,td00,chi1)
# 0.7512627811178455 10.128187149098038 0.8243797320063613 0.4797233828564622 0.018704566649964942
#x0f,s0f,y0f,td0f,chi2
 # main(s0,x0,y0,td)
#matrix,p11,p22,p33,c1,c2,c3,c4=main(4.898902304319312,3.9772231249928223/2/pi ,0.4691503944650351/2/pi,0.055216417596723535)
#matrix,p11,p22,p33,c1,c2,c3,c4=main(8.461348006,0.7340 ,0.772464685,0.54919)
#matrix,p1,p2,p3,c1,c2,c3,c4=main(10.52838666,0.61037,0.0908695,0.7265)


#graph(2,0.8,0.6,0.01)


def algorithme_genetique(data) :
      
    
    parent=10 # nombre de parent 
    #paramètre de départ si connu
    x0f2=[0.7340]
    y0f2=[0.11]
    td0f2= [0.54919]
    s0f2=[1]
    taille=400
    chi2=10000
    for iteration4real in range(1,10000) :
        mutation=5/iteration4real # taux de mutation en pourcentage adaptatif (largeur a mi-hauteur gaussienne)
      
        xf=np.zeros((taille))
        yf=np.zeros((taille))
        tdf=np.zeros((taille))
        s0f=np.zeros((taille))
        chif=np.zeros((taille))
        
        for i in range(0,len(x0f2)):
            x00=x0f2[i]+np.random.normal(x0f2[i],x0f2[i]*mutation)
            y00=y0f2[i]+np.random.normal(x0f2[i],x0f2[i]*mutation)
            td00=td0f2[i]+np.random.normal(x0f2[i],x0f2[i]*mutation)
            s00=s0f2[i]+np.random.normal(x0f2[i],x0f2[i]*mutation)
            chi1,x00,y00,td00,s00=boucle_genetique(xf,yf,tdf,s0f,chif,data,taille,iteration4real,x00,y00,td00,s00,parent)
            
            if np.min(chi1)<chi2 :
                chi2=np.min(chi1)
                popt=np.where(chi1==chi2)[0]
                if len(popt)!=1 :
                    popt=popt[0]
               
                x0f2,s0f2,y0f2,td0f2=x00,s00,y00,td00
                print(x00[popt],s00[popt],y00[popt],td00[popt])
                #graph(1,x0f2,y0f2,td0f2)
                
    return x0f2,s0f2,y0f2,td0f2,chi2

#@njit(parallel=True)
def boucle_genetique(xf,yf,tdf,s0f,chif,data,taille,iteration4real,x00,y00,td00,s00,parent) :
        for iteration in prange(0,taille) :
          
            
            
            #print(iteration,iteration4real)
            
            s0test,x0test,y0test,tdtest=seed(x00,y00,td00,s00,iteration4real)
            
            matrix,p1,p2,p3,c1,c2,c3,c4=main(s0test,x0test,y0test,tdtest)
            print((1/len(p1)*np.sum((p1-data[:,1]))**2)+(1/len(p1)*np.sum((p2-data[:,2]))**2)+(1/len(p1)*np.sum((p3-data[:,3])**2)))
        
            chif[iteration]=(1/len(p1)*np.sum((p1-data[:,1]))**2)+(1/len(p1)*np.sum((p2-data[:,2]))**2)+(1/len(p1)*np.sum((p3-data[:,3])**2))
        
            xf[iteration]=x0test
            yf[iteration]=y0test
            tdf[iteration]=tdtest
            s0f[iteration]=s0test
            
            
        
        print(chif)
        chi1=np.sort(chif)
        chi1=chi1[0:parent]
        
        x00=[]
        y00=[]
        td00=[]
        s00=[]
        for i in chi1 :
            #boucle pour créer les enfants
            popt=np.where(chif==i)[0]
            
            x00.append(np.mean([xf[popt][0],xf[np.random.randint(0,len(xf)-1)]]))
            y00.append(np.mean([yf[popt][0],yf[np.random.randint(0,len(yf)-1)]]))
            td00.append(np.mean([tdf[popt][0],tdf[np.random.randint(0,len(tdf)-1)]]))
            s00.append(np.mean([s0f[popt][0],s0f[np.random.randint(0,len(s0f)-1)]]))
            
        return chi1,x00,y00,td00,s00

x00,s00,y00,td00,chi1=algorithme_genetique(data)
print(x00,s00,y00,td00,chi1)