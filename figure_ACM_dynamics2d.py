#! /usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import time
import multiprocessing

fontsize=16
plt.rc("font",size=fontsize)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

color=['#ff0000','#ff6600','#00ff00','#006600','#00ffff','#0000ff','#cc66ff','k']
fmt='os>*^hv'

clock=time.time()

beta=2
init=0

Q=[4,4,8,8]
EPSILON=[0.2,0.8,0.9,0.9]
RHO0=[1.5,1.5,1.5,2]
LX=[400,400,800,800]
LY=[50,50,20,20]
RAN=[1,0,0,0]

TMAX=[2000000,2000000,4000000,4000000]
DT=[500,500,1000,1000]

DeltaT=1./(1.+np.exp(2*beta))
dpi=240

multi=True
movie=False
NCPU=4

for arg in sys.argv[1:]:
	if "-NCPU=" in arg:
		NCPU=int(arg[6:])
	elif "-movie" in arg:
		movie=True
	else:
		print("Bad Argument: ",arg)
		sys.exit(1)
		
if NCPU==1:
	multi=False
elif NCPU<1:
	print("Bad value of NCPU: ",NCPU)
	sys.exit(1)


norm=matplotlib.colors.Normalize(vmin=-np.pi,vmax=np.pi)

def Snapshot(i):
	if not os.path.isfile('snapshots/figure_ACM_theta_%d.png'%(i)):
		fig=plt.figure(figsize=(8,6))
		gs=matplotlib.gridspec.GridSpec(4,1,width_ratios=[1],height_ratios=[1,1,1,1],left=0.045,right=1.075, bottom=0.045,top=0.96,hspace=0.5,wspace=0.1)
		
		for k in [0,1,2,3]:
			ax=plt.subplot(gs[k,0])
			t=i*DT[k]
			
			data=np.fromfile('data_ACM_particles/ACM_particles_q=%d_beta=%.8g_epsilon=%.8g_rho0=%.8g_LX=%d_LY=%d_init=%d_ran=%d_t=%d.bin'%(Q[k],beta,EPSILON[k],RHO0[k],LX[k],LY[k],init,RAN[k],t),dtype=np.float32).reshape(-1, 3)
			
			X,Y,THETA=data.T
			THETA=(THETA+np.pi)%(2*np.pi)-np.pi
			Npart=len(data)
			
			q=ax.scatter(X,Y,c=THETA,cmap='hsv',norm=norm,s=1,marker='o',linewidths=0,edgecolors='none',rasterized=True)
			cb=plt.colorbar(q,ticks=[-np.pi,0,np.pi],pad=0.01,aspect=5)
			cb.ax.set_yticklabels(['$-\pi$','$0$','$\pi$'])
			cb.solids.set_rasterized(True)
			
			#plt.axis('equal')
			plt.xlim([0,LX[k]])
			plt.ylim([0,LY[k]])
			plt.xticks([0,0.25*LX[k],0.5*LX[k],0.75*LX[k],LX[k]])
			plt.yticks([0,0.5*LY[k],LY[k]])
			ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('$%.8g$'))
			ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('$%.8g$'))
			
			plt.text(0.01*LX[k],0.01*LY[k],'$t=%d$'%(t*DeltaT),ha='left',va='bottom',fontsize=16,bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),zorder=1)
			plt.text(LX[k],1.1*LY[k],'$q=%d$, $\\beta=%.8g$, $\\varepsilon=%.8g$, $\\rho_0=%.8g$'%(Q[k],beta,EPSILON[k],RHO0[k]),ha='right',va='center',fontsize=16)		
		
		plt.savefig('snapshots/figure_ACM_theta_%d.png'%(i),dpi=dpi)
		plt.close()

		print('%d/%d -t=%d -tcpu=%d'%(i+1,Nsnaps,t,time.time()-clock))
		del data,fig
	
os.system('mkdir -p snapshots')

ARG=range(TMAX[0]//DT[0]+1)
Nsnaps=len(ARG)
print('%d Snapshots'%Nsnaps)

if multi:
	pool=multiprocessing.Pool(NCPU)
	results=pool.imap_unordered(Snapshot,ARG)
	pool.close()
	pool.join()
else:
	for i in ARG:
		Snapshot(i)
		
if movie:
	os.system('mkdir -p movies')	
	os.system('ffmpeg -v quiet -stats -y -r 25/1 -i snapshots/figure_ACM_theta_%%01d.png -c:v h264 -r 25 -s %dx%d -crf 30 movies/movie_ACM_theta.mp4'%(8*dpi,6*dpi))
	if Nsnaps>tmax//DT[k]:
		os.system('rm snapshots/figure_ACM_theta_*.png')
		
	
print('OK - time=%d sec'%(time.time()-clock))
