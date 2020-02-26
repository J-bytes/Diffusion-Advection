# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:19:04 2020

@author: joeda
"""

#Importation des fonctions
import numpy as np
from numba import njit,prange
from scipy.constants import pi
import matplotlib.pyplot as plt
import time
from matplotlib import cm


data=np.loadtxt('data.txt')


@njit(cache=True)
def Terme_Source(xx,yy,tt,S0,x0,y0): #équation 2.85
    
    
    w=0.25
    r2=(xx-x0)**2+(yy-y0)**2
    S=S0*np.exp(-r2/w**2)
    return S

############-Périodicité-###############
@njit(cache=True)
def periodic(f,N):
    
    f[1:N+1,0]=f[1:N+1,N]
    f[1:N+1,N+1]=f[1:N+1,1]
    f[0,1:N+1]= f[N,1:N+1]
    f[N+1,1:N+1]=f[1,1:N+1]
    f[0,0]=f[N-1,N-1]
    f[N+1,N+1]=f[2,2]
    f[0,N+1]=f[N-1,2]
    f[N+1,0]=f[2,N-1]
    return f
   
#########Vitesse#########
@njit(cache=True)
def vitesse(x,y,t,N,A,eps,omega) :

        ux1=A*np.cos(y+eps*np.sin(omega*t))
        
        uy1=A*np.sin(x+eps*np.cos(omega*t))
        
        ux=np.ones((N,N))*ux1
        
        uy=np.ones((N,N))*uy1
        
        return(ux,uy)

#########Advection Lax Wendroff############
@njit(cache=True)
def advectif_Lax_Wendroff(c,N,dt,dx,D,t,x,y,A,eps,omega):
    
    uxkp,uyjp=vitesse(x+dx/2,y+dx/2,t+dt/2,N,A,eps,omega)
    uxkm,uyjm=vitesse(x-dx/2,y-dx/2,t+dt/2,N,A,eps,omega)
    """
    Fx11,Fy11=Fcalc(c,N,1,1,dx,dt,uxkp,uyjp,D,t)
    Fx12,Fy12=Fcalc(c,N,1,-1,dx,dt,uxkp,uyjm,D,t)
    Fx21,Fy21=Fcalc(c,N,-1,1,dx,dt,uxkm,uyjp,D,t)
    Fx22,Fy22=Fcalc(c,N,-1,-1,dx,dt,uxkm,uyjm,D,t)
    """
    c_demi11 = 1/4*(c[2:N+2,1:N+1] + c[1:N+1,1:N+1] + c[2:N+2,2:N+2] + c[1:N+1,2:N+2]) - dt/2*( uxkp*(c[2:N+2,1:N+1] - c[1:N+1,1:N+1])/dx + uyjp*(c[1:N+1,2:N+2] - c[1:N+1,1:N+1])/dx)  #j+1, k+1 Éqn (2.78)
    c_demi21 = 1/4*(c[0:N,1:N+1] + c[1:N+1,1:N+1] + c[0:N,2:N+2] + c[1:N+1,2:N+2]) - dt/2*( -uxkp*(c[0:N,1:N+1] - c[1:N+1,1:N+1])/dx + uyjm*(c[1:N+1,2:N+2] - c[1:N+1,1:N+1])/dx)  #j-1, k+1
    c_demi12 = 1/4*(c[2:N+2,1:N+1] + c[1:N+1,1:N+1] + c[2:N+2,0:N] + c[1:N+1,0:N]) - dt/2*( uxkm*(c[2:N+2,1:N+1] - c[1:N+1,1:N+1])/dx - uyjp*(c[1:N+1,0:N] - c[1:N+1,1:N+1])/dx)  #j+1, k-1 
    c_demi22 = 1/4*(c[0:N,1:N+1] + c[1:N+1,1:N+1] + c[0:N,0:N] + c[1:N+1,0:N]) - dt/2*( -uxkm*(c[0:N,1:N+1] - c[1:N+1,1:N+1])/dx - uyjm*(c[1:N+1,0:N] - c[1:N+1,1:N+1])/dx)  #j-1, k-1
    
    Fx11 = uxkp*c_demi11 # j+1/2, k+1/2
    Fx21 = uxkp*c_demi21 # j-1/2, k+1/2
    Fx12 = uxkm*c_demi12 # j+1/2, k-1/2
    Fx22 = uxkm*c_demi22 # j-1/2, k-1/2

    Fy11 = uyjp*c_demi11 # j+1/2, k+1/2
    Fy21 = uyjm*c_demi21 # j-1/2, k+1/2
    Fy12 = uyjp*c_demi12 # j+1/2, k-1/2
    Fy22 = uyjm*c_demi22 # j-1/2, k-1/2     
    
    #F = -1/2*dt/dx*(Fx12 - Fx22 + Fx11 - Fx21) - 1/2*dt/dx*(Fy11 - Fy12 + Fy21 - Fy22)
    #return F    
    
                
    return  (-dt/2/dx*(Fx11+Fx12-Fx21-Fx22)-dt/2/dx*(Fy11-Fy12+Fy21-Fy22))
   
   
#########Advection Leapfrog###########
@njit(cache=True)
def advectif_leapfrog(c,N,D,dt,dx,ux,uy) :
    return -dt*(ux[::2,::2][1:N+1,1:N+1]*(c[2:N+2,1:N+1] - c[0:N,1:N+1])/dx + uy[::2,::2][1:N+1,1:N+1]*(c[1:N+1,2:N+2] - c[1:N+1,0:N])/dx )
                   
  
#########Diffusion Leapfrog##########
@njit(cache=True)
def diffusif_leapfrog(C,N,D,dt,dxlax):
    dylax=dxlax
    return (D*dt)*(dxlax**(-2)*(C[2:,1:N+1]-2*C[1:N+1,1:N+1]+C[0:N,1:N+1])+(dylax**(-2))*(C[1:N+1,2:]-2*C[1:N+1,1:N+1]+C[1:N+1,0:N]))
##########-Principal-###############   
@njit(cache=True)
def main(s0,x0,y0,td,A,tf) :
    
    #####Paramètre de maille###########
    N=401
    dt=1e-3
    dx=2*pi/(N-1)
    x0,y0=x0*(N-1)*dx,y0*(N-1)*dx
    tmax=4
    NITER=int(tmax/dt)+1
    
    
    
    #####-Paramètre de diffusion-##########
    #A=-np.sqrt(6)
    
    w=0.25#source
    omega=5#vitesse
    
    
    #td=0.01
    #tf=2.5
    eps=1  #?????????
    
    Pe=1000
    D=1/Pe
    
    ######-Boucle d'iteration-##########
    c=np.zeros((N+2,N+2))
    cnp1=np.zeros((N+2,N+2))
   
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
 
    
    for iteration in range(0,NITER) :
        #print(iteration)
        t=iteration*dt
       # print(np.max(c),t)
        
        if td<=t<=tf :
            cnp1[1:N+1,1:N+1]= dt*Terme_Source(x,y,t,s0,x0,y0)+advectif_Lax_Wendroff(c,N,dt,dx,D,t,x,y,A,eps,omega)+c[1:N+1,1:N+1]+diffusif_leapfrog(c,N,D,dt,dx)
        else :
            cnp1[1:N+1,1:N+1]=advectif_Lax_Wendroff(c,N,dt,dx,D,t,x,y,A,eps,omega)+c[1:N+1,1:N+1]+diffusif_leapfrog(c,N,D,dt,dx)
       
       
               
        c=np.copy(periodic(cnp1,N))
        point1[iteration]=c[25*4+1][75*4+1]
        point2[iteration]=c[50*4+1][50*4+1]
        point3[iteration]=c[75*4+1][25*4+1]
       
        if round(t,3)==0.5 : 
            c1=np.copy(c[1:N])
            #print(iteration)
        if round(t,3)==1 : 
            c2=np.copy(c[1:N])
            #print(iteration)
        if round(t,3)==2 : 
            c3=np.copy(c[1:N])
            #print(iteration)
        if round(t,3)==3 : 
            c4=np.copy(c[1:N])
            #print(iteration)
        
    return   c[1:N+1,1:N+1],point1,point2,point3,c1,c2,c3,c4
    

@njit(parallel=True)
def boucle(xf,yf,tdf,s0f,chif,data,taille,iteration4real,x00,y00,td00,s00) :
        for iteration in prange(0,taille) :
            print(iteration,iteration4real)
            
            s0test,x0test,y0test,tdtest=seed(x00,y00,td00,s00,iteration4real)
            matrix,p1,p2,p3,c1,c2,c3,c4=main(s0test,x0test,y0test,tdtest,-np.sqrt(6),2.5)
                     
            chif[iteration]=np.sum(1/len(p1)*(p1/np.max(p1)-data[:,1]/np.max(data[:,1]))**2)+np.sum(1/len(p1)*(p2/np.max(p2)-data[:,2]/np.max(data[:,2]))**2)+np.sum(1/len(p1)*(p3/np.max(p3)-data[:,3]/np.max(data[:,3]))**2)
        
            xf[iteration]=x0test
            yf[iteration]=y0test
            tdf[iteration]=tdtest
            s0f[iteration]=s0test
        
        chi1=np.min(chif)
        popt=np.where(chif==chi1)[0]
        
        x00,y00,td00,s00=xf[popt][0],yf[popt][0],tdf[popt][0],s0f[popt][0]
        
        return chi1,x00,y00,td00,s00
    
    
def optimisation(data) :
    # 0.7512627811178455 10.128187149098038 0.8243797320063613 0.4797233828564622 0.018704566649964942
    #x0f,s0f,y0f,td0f,chi2
    # main(s0,x0,y0,td)
    x00=0.7340
    y00=0.772464685
    td00= 0.54919
    s00=1
    taille=40
    chi2=10000
    for iteration4real in range(8,18) :
        xf=np.zeros((taille))
        yf=np.zeros((taille))
        tdf=np.zeros((taille))
        s0f=np.zeros((taille))
        chif=np.zeros((taille))
        chi1,x00,y00,td00,s00=boucle(xf,yf,tdf,s0f,chif,data,taille,iteration4real,x00,y00,td00,s00)
        if chi1<chi2 :
            chi2=chi1
            x0f2,s0f2,y0f2,td0f2=x00,s00,y00,td00
            print(x0f2,s0f2,y0f2,td0f2,chi2)
                
    return x0f2,s0f2,y0f2,td0f2,chi2
@njit
def seed(x0,y0,td,s0,iteration) :
         if iteration==0 :
             x0test=np.random.uniform(0,1)
             y0test=np.random.uniform(0,1)
             tdtest=np.random.uniform(0,0.5)
             s0test=np.random.uniform(0,10)
        
         else :
             s0max=s0/iteration
             x0max=x0/iteration
             y0max=y0/iteration
             tdmax=td/iteration
             x0test=np.random.normal(x0,x0max)
             y0test=np.random.normal(y0,y0max)
             tdtest=np.random.normal(td,tdmax)
             s0test=np.random.normal(s0,s0max)
        
         return s0test,x0test,y0test,tdtest

data=np.loadtxt('data.txt')
#x00,s00,y00,td00,chi1=optimisation(data)
#print(x00,s00,y00,td00,chi1)
# 0.7512627811178455 10.128187149098038 0.8243797320063613 0.4797233828564622 0.018704566649964942
#0.5176943000126841 15.509481752119644 0.060712788008025646 0.6592223103761399 0.17235407766748773
#0.7569501698791052 11.142174767212207 0.8274971251851719 0.3288901382065027 0.014165935132421204
#0.7495399108413119 8.65305752311657 0.8073222303055961 0.4076088972618495 0.015782210820716028
#0.7398400989103205 7.72337335606018 0.7873354550706197 0.4552390765863496 0.019717813702397807

#x0f,s0f,y0f,td0f,chi2
 # main(s0,x0,y0,td)
"""
j'ai  td = 0.055216417596723535 ;
 S0 = 4.898902304319312 ; 
 x0 = 3.9772231249928223 ; 
 y0 = 0.4691503944650351 à date
"""

#0.7813737326292098 -0.8903635623851276 -0.3564058724805934 0.6204313130050954 0.17235407790651924

#0.8489757596334115 1.7779864465638542 0.816249825316977 0.5034985647696626 0.17235402451637877
#matrix,p11,p22,p33,c1,c2,c3,c4=main(-0.8903635623851276,0.7813737326292098 ,-0.3564058724805934,0.6204313130050954)
#matrix,p11,p22,p33,c1,c2,c3,c4=main(2,0.8 ,0.6,0.01,np.sqrt(6),1.75)

#matrix,p11,p22,p33,c1,c2,c3,c4=main(8.461348006,0.7340 ,0.772464685,0.54919,-np.sqrt(6),2.5)

def graph(s0,x0,y0,td,A,tf) :   
    matrix,p11,p22,p33,c1,c2,c3,c4=main(s0,x0,y0,td,A,tf)
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


#graph(2,0.8,0.6,0.01,A=np.sqrt(6),tf=1.75)
    


def algorithme_genetique(data) :
      
    
    parent=2 # nombre de parent 
    #paramètre de départ si connu
    x0f2=[0.7340]
    y0f2=[0.11]
    td0f2= [0.54919]
    s0f2=[1]
    taille=20
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
               
                x0f2,s0f2,y0f2,td0f2=x00,s00,y00,td00
                print(x00[popt],s00[popt],y00[popt],td00[popt])
                #graph(1,x0f2,y0f2,td0f2)
                
    return x0f2,s0f2,y0f2,td0f2,chi2

#@njit(parallel=True)
def boucle_genetique(xf,yf,tdf,s0f,chif,data,taille,iteration4real,x00,y00,td00,s00,parent) :
        for iteration in prange(0,taille) :
          
            
            
            #print(iteration,iteration4real)
            
            s0test,x0test,y0test,tdtest=seed(x00,y00,td00,s00,iteration4real)
            
            matrix,p1,p2,p3,c1,c2,c3,c4=main(s0test,x0test,y0test,tdtest,-np.sqrt(6),2.5)
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

#x00,s00,y00,td00,chi1=algorithme_genetique(data)
#print(x00,s00,y00,td00,chi1)    
