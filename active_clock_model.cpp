/*C++ CODE - q-STATE ACM SIMULATION - WRITTEN BY M. MANGEAT (2026)*/

//////////////////////
///// LIBRAIRIES /////
//////////////////////

//Public librairies.
#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <string.h>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>

using namespace std;

//Personal libraries.
#include "lib/random_OMP.cpp"
#include "lib/special_functions.cpp"

///////////////////////////////
///// CLASS FOR ACM SPINS /////
///////////////////////////////

class ACMParticle
{
	public:
	
	double x, y; //position.
	double xnew, ynew; //intermediate position.
	int sigma; //state.
	
	ACMParticle(const int &init, const int &q, const int &LX, const int &LY);
	void hop_guess(const double &r, const double &epsilon, const int &q, const int &LX, const int &LY, const vector<double> &COS, const vector<double> &SIN);
	void hop_apply();
	void flip(const int &sigma_new);
};

//Creation of the spin.
ACMParticle::ACMParticle(const int &init, const int &q, const int &LX, const int &LY)
{
	//Initial state.
	if (init==0)
	{
		//Random initial position + random q-state orientation.
		x=ran()*LX;
		y=ran()*LY;
		sigma=int(q*ran());
	}
	else if (init==1)
	{
		//Random initial position + ordered orientation.
		x=ran()*LX;
		y=ran()*LY;
		sigma=0;
	}
	else if (init==2)
	{
		//Initial transverse band.
		x=(2+ran())*0.2*LX;
		y=ran()*LY;
		sigma=0;
	}
	else if (init==3)
	{
		//Initial longitudinal lane (q must be in 4N).
		x=(2+ran())*0.2*LX;
		y=ran()*LY;
		sigma=q/4;
	}
	else
	{
		cerr << "BAD INIT VALUE: " << init << endl;
		abort();
	}
}

//Hop to the direction theta and implement the periodicity LX/LY.
void ACMParticle::hop_guess(const double &r, const double &epsilon, const int &q, const int &LX, const int &LY, const vector<double> &COS, const vector<double> &SIN)
{
	//Move to the favoured direction.
	int d=sigma;
	if (r>epsilon) //Remark: epsilon=0 the direction is purely random and epsilon=1 the direction is purely the favoured.
	{
		d=int(q*ran());
	}
	
	//Update intermediate positions.		
	xnew=x+COS[d];
	ynew=y+SIN[d];
	
	//Periodicity.
	if (xnew<0) xnew+=LX;
	if (xnew>=LX) xnew-=LX;
	if (ynew<0) ynew+=LY;
	if (ynew>=LY) ynew-=LY;
}

//Apply the hopping.
void ACMParticle::hop_apply()
{
	x=xnew;
	y=ynew;
}

//Flip to the new state.
void ACMParticle::flip(const int &sigma_new)
{
	sigma=sigma_new;
}

/////////////////////////////////////////////////////
///// ARRAYS OF INDICES OF PARTICLES BY SECTORS /////
/////////////////////////////////////////////////////

class Sectors
{
	vector< vector<int> > sec;
	
	public:
	
	Sectors(const int &Ngrid);
	const vector<int>& get(const int &k) const;
	void add(const int &k, const int &index);
	void remove(const int &k, const int &index);
};

//Creation of the array.
Sectors::Sectors(const int &Ngrid)
{
	sec=vector< vector<int> >(Ngrid, vector<int>(0));
}

//Return the indexes of sector k.
const vector<int>& Sectors::get(const int &k) const
{
	return sec[k];
}

//Add one particle to the sector k.
void Sectors::add(const int &k, const int &index)
{
	sec[k].push_back(index);
}

//Remove one particle from the sector k.
void Sectors::remove(const int &k, const int &index)
{
	for (int l=0; l<sec[k].size(); l++)
	{
		if (sec[k][l]==index)
		{
			sec[k].erase(sec[k].begin()+l);
			return;
		}
	}
	
	cerr << "ERROR: particle " << index << " not found in sector " << k << endl;
	abort();
}

////////////////////////////////
///// AVERAGES IN SUBBOXES /////
////////////////////////////////

class averages
{
	public:
	
	int L0, rx, ry; //parameters of the boxes.
	vector<double> n, n2; //number fluctuations.
	vector<double> m, m2; //magnetization fluctuations.
	int Nav; //number of averages (time + space).
	
	averages(const int &LX, const int &LY);
	void update(const vector<int> &RHO, const vector<double> &MX, const vector<double> &MY, const int &LX, const int &LY);
	void exportFile(const double &beta, const double &epsilon, const double &rho0, const int &q, const int &LX, const int &LY, const int &init, const int &RAN);
};

//Creation of the averaged quantities.
averages::averages(const int &LX, const int &LY)
{
	L0=min(LX,LY);
	rx=max(1,LX/LY);
	ry=max(1,LY/LX);

	n=vector<double>(L0,0.);
	n2=vector<double>(L0,0.);
	m=vector<double>(L0,0.);
	m2=vector<double>(L0,0.);
	
	Nav=0;
}

//Update the averaged quantities at time t.
void averages::update(const vector<int> &RHO, const vector<double> &MX, const vector<double> &MY, const int &LX, const int &LY)
{
	static const int Nboxes=10;
	//Averages on all sub-boxes.
	#pragma omp parallel for default(shared)
	for (int l=1; l<L0; l++)
	{
		//Select Nboxes different sub-boxes of size l.
		for (int nbox=0; nbox<Nboxes; nbox++)
		{
			//Random position for the bottom left corner.
			const int x0=int((LX-rx*l)*ran()),  y0=int((LY-ry*l)*ran());
			
			//Density and magnetization in this sub-box.
			long unsigned int rho=0;
			double mx=0., my=0.;
			
			for (int x=x0; x<x0+rx*l; x++)
			{
				for (int y=y0; y<y0+ry*l; y++)
				{
					const int k=x*LY+y;
					rho+=RHO[k];
					mx+=MX[k];
					my+=MY[k];
				}
			}
			
			const double mag2=mx*mx+my*my;
			
			//Add to n, n2, m, m2.
			n[l]+=double(rho)/Nboxes;
			n2[l]+=double(rho*rho)/Nboxes;
			m[l]+=sqrt(mag2)/Nboxes;
			m2[l]+=mag2/Nboxes;
		}
	}
	//Increase the number of averages.
	Nav++;
}


//Export averages in a file.
void averages::exportFile(const double &beta, const double &epsilon, const double &rho0, const int &q, const int &LX, const int &LY, const int &init, const int &RAN)
{
	static const int returnSystem=system("mkdir -p data_ACM_averages/");
	stringstream ss;	
	ss << "./data_ACM_averages/ACM_fluctuations_q=" << q << "_beta=" << beta << "_epsilon=" << epsilon << "_rho0=" << rho0 << "_LX=" << LX << "_LY=" << LY << "_init=" << init << "_ran=" << RAN << ".txt";	
	string nameAV = ss.str();			
	ofstream fileAV(nameAV.c_str(),ios::trunc);	
	fileAV.precision(6);
	
	fileAV << "#The number of averages are: " << Nav << endl;

	for (int l=1; l<L0; l++)
	{
		fileAV << l << " " << n[l]/Nav << " " << n2[l]/Nav-square(n[l]/Nav) << " " << m[l]/Nav << " " << m2[l]/Nav-square(m[l]/Nav) << endl;
	}
	
	fileAV.close();
}


/////////////////////
///// FONCTIONS /////
/////////////////////

//Total average on the whole space.
template <typename T>
double average(const vector<T> &RHO, const int &Ngrid)
{
	double rhoAv=0;
	for (int k=0; k<Ngrid; k++)
	{
		rhoAv+=RHO[k];
	}
	return double(rhoAv)/Ngrid;
}

//Distance between two particles in the periodic domain.
double distance2(const ACMParticle &part1, const ACMParticle &part2, const int &LX, const int &LY)
{
	const double DX=fabs(part1.x-part2.x);
	const double DY=fabs(part1.y-part2.y);	
	return square(min(DX,LX-DX)) + square(min(DY,LY-DY));
}

//Export the density and the mean orientation in files.
void exportDynamics(const vector<int> &RHO, const vector<double> &MX, const vector<double> &MY, const double &beta, const double &epsilon, const double &rho0, const int &q, const int &LX, const int &LY, const int &init, const int &RAN, const int &t)
{
	const int Nsectors=LX*LY;
	vector<int16_t> rho(Nsectors,0);
	vector<float> theta(Nsectors,0);
	
	for (int Y0=0; Y0<LY; Y0++)
	{
		for (int X0=0; X0<LX; X0++)
		{
			rho[Y0*LX+X0]=RHO[X0*LY+Y0];
			theta[Y0*LX+X0]=atan2(MY[X0*LY+Y0],MX[X0*LY+Y0]);
		}
	}

	static const int returnSystem=system("mkdir -p data_ACM_dynamics2d/");
	stringstream ss1,ss2;
	
	ss1 << "./data_ACM_dynamics2d/ACM_RHO_q=" << q << "_beta=" << beta << "_epsilon=" << epsilon << "_rho0=" << rho0 << "_LX=" << LX << "_LY=" << LY << "_init=" << init << "_ran=" << RAN << "_t=" << t << ".bin";	
	string nameRHO = ss1.str();			
	ofstream fileRHO(nameRHO.c_str(),ios::binary);	
	fileRHO.write(reinterpret_cast<const char*>(rho.data()), Nsectors*sizeof(int16_t));
	fileRHO.close();
	
	ss2 << "./data_ACM_dynamics2d/ACM_THETA_q=" << q << "_beta=" << beta << "_epsilon=" << epsilon << "_rho0=" << rho0 << "_LX=" << LX << "_LY=" << LY << "_init=" << init << "_ran=" << RAN << "_t=" << t << ".bin";
	string nameTHETA = ss2.str();	
	ofstream fileTHETA(nameTHETA.c_str(),ios::binary);
	fileTHETA.write(reinterpret_cast<const char*>(theta.data()), Nsectors*sizeof(float));
	fileTHETA.close();
}

//Export particle positions and orientations in a file.
void exportParticles(const double &beta, const double &epsilon, const double &rho0, const int &q, const int &LX, const int &LY, const int &init, const int &RAN, const int &t, const int &Npart, const vector<ACMParticle> &PART)
{
	//Creation of the file.
	static const int dossier=system("mkdir -p ./data_ACM_particles/");
	stringstream ssPart;
	ssPart << "./data_ACM_particles/ACM_particles_q=" << q << "_beta=" << beta << "_epsilon=" << epsilon << "_rho0=" << rho0 << "_LX=" << LX << "_LY=" << LY << "_init=" << init << "_ran=" << RAN << "_t=" << t << ".bin";
	string namePart = ssPart.str();
	ofstream filePart(namePart.c_str(),ios::binary);
	
	const float dphi=static_cast<float>(2.0*M_PI/q);
	
	//Write in the file.
	for (int i=0;i<Npart;i++)
	{
		const float phi=static_cast<float>(dphi*PART[i].sigma);
		const float part[3]={static_cast<float>(PART[i].x),static_cast<float>(PART[i].y),phi};
		filePart.write(reinterpret_cast<const char*>(part), 3*sizeof(float));
	}
	filePart.close();
}

///////////////////////////////////////
///// READ COMMAND LINE ARGUMENTS /////
///////////////////////////////////////

void ReadCommandLine(int argc, char** argv, int &q, double &beta, double &rho0, double &epsilon, int &LX, int &LY, int &tmax, int& init, int &RAN, int &THREAD_NUM)
{
 	for( int i = 1; i<argc; i++ )
	{
		if (strstr(argv[i], "-q=" ))
		{
			q=atoi(argv[i]+3);
		}
		else if (strstr(argv[i], "-beta=" ))
		{
			beta=atof(argv[i]+6);
		}
		else if (strstr(argv[i], "-rho0=" ))
		{
			rho0=atof(argv[i]+6);
		}
		else if (strstr(argv[i], "-epsilon=" ))
		{
			epsilon=atof(argv[i]+9);
		}
		else if (strstr(argv[i], "-LX=" ))
		{
			LX=atoi(argv[i]+4);
		}
		else if (strstr(argv[i], "-LY=" ))
		{
			LY=atoi(argv[i]+4);
		}
		else if (strstr(argv[i], "-tmax=" ))
		{
			tmax=atoi(argv[i]+6);
		}
		else if (strstr(argv[i], "-init=" ))
		{
			init=atoi(argv[i]+6);
		}
		else if (strstr(argv[i], "-ran=" ))
		{
			RAN=atoi(argv[i]+5);
		}
		else if (strstr(argv[i], "-threads=" ))
		{
			THREAD_NUM=atoi(argv[i]+9);
		}
		else
		{
			cerr << "BAD ARGUMENT : " << argv[i] << endl;
			abort();
		}
	}
	cout << "Parameters: -q=" << q << " -beta=" << beta << " -rho0=" << rho0 << " -epsilon=" << epsilon << " -LX=" << LX << " -LY=" << LY << " -tmax=" << tmax << " -init=" << init << " -ran=" << RAN << " -threads=" << THREAD_NUM << endl;
}

/////////////////////
///// MAIN CODE /////
/////////////////////

int main(int argc, char *argv[])
{
	//Physical parameters: beta=inverse temperature, rho0=average density, epsilon=self-propulsion, LX*LY=size of the box, q=number of states.
	const double D0=1.;
	double beta=2., rho0=1.5, epsilon=0.2;
	int LX=400, LY=50, q=4;
	
	//Numerical parameters: tmax=maximal time, init=initial condition, THREAD_NUM=number of threads used in OpenMP.
	int tmax=1000000, init=0, RAN=0, THREAD_NUM=4;

	//Parameters in arguments.
	ReadCommandLine(argc,argv,q,beta,rho0,epsilon,LX,LY,tmax,init,RAN,THREAD_NUM);
	
	//Total number of particles.
	int Npart=int(LX*LY*rho0);
	
	/*if (Npart%THREAD_NUM!=0)
	{
		cerr << "BECAREFUL SOME PARTICLES ARE NOT MOVING BECAUSE THE NUMBER OF THREADS (" <<  THREAD_NUM << ") IS NOT DIVIDING THE NUMBER OF PARTICLES (" << Npart << ")!!!" << endl;
		abort();
	}*/
	if (epsilon<0. or epsilon>1.) abort();
	
	//OpenMP threads creation.
	omp_set_dynamic(0);
	omp_set_num_threads(THREAD_NUM);
	cout << OMP_MAX_THREADS << " maximum threads on this node. " << THREAD_NUM << " threads will be used." << endl;

	//Start the random number generator.
	init_gsl_ran();
	for (int k=0; k<THREAD_NUM; k++)
	{
		gsl_rng_set(GSL_r[k],THREAD_NUM*RAN+k);
	}
	
	//Presave cosine and sine values of the discrete angles.
	vector<double> COS(q), SIN(q), COS_DIFF(q*q);

	for (int s=0; s<q; s++)
	{
		const double phi=2*M_PI*s/q;
		COS[s]=cos(phi);
		SIN[s]=sin(phi);
	}

	for (int s1=0; s1<q; s1++)
	{
		for (int s2=0; s2<q; s2++)
		{
			COS_DIFF[s1*q+s2]=COS[s1]*COS[s2]+SIN[s1]*SIN[s2]; //cos(phi1-phi2)
		}
	}

	//Creation of sectors of size 1x1.
	Sectors SEC(LX*LY);
	
	//Define the particle density and magnetization.
	vector<int> RHO(LX*LY,0);
	vector<double> MX(LX*LY,0.), MY(LX*LY,0.);
	
	//Creation of Npart particles.
	vector<ACMParticle> PART;
	PART.reserve(Npart);
	
	for (int i=0;i<Npart;i++)
	{
		ACMParticle part(init,q,LX,LY);
		
		const int k0=int(part.x)*LY+int(part.y), SIGMA=part.sigma;
		
		SEC.add(k0,i);
		RHO[k0]++;
		MX[k0]+=COS[SIGMA];
		MY[k0]+=SIN[SIGMA];
		
		PART.push_back(part);
	}
	
	//File for averages.
	const int returnSystem=system("mkdir -p data_ACM_averages/");
	stringstream strAVERAGES;
	strAVERAGES << "./data_ACM_averages/ACM_AVERAGES_q=" << q << "_beta=" << beta << "_epsilon=" << epsilon << "_rho0=" << rho0 << "_LX=" << LX << "_LY=" << LY << "_init=" << init << "_ran=" << RAN << ".txt";
	string nameAVERAGES = strAVERAGES.str();			
	ofstream fileAVERAGES(nameAVERAGES.c_str(),ios::trunc);
	fileAVERAGES.precision(6);
	
	//Fluctuations.
	averages AV(LX,LY);
	double teq=5000;
	
	//Time and angle increment.
	const double DeltaT=1./(D0+exp(2*beta));
	
	//Get the probability to hop (constant over the time).
	const double proba_hop=D0*DeltaT;
	
	//Number of particles per threads.
	const int Nth=Npart/THREAD_NUM, Nextra=Npart%THREAD_NUM;
	vector<int> ITH(THREAD_NUM+1,0);
	
	for (int k=0; k<THREAD_NUM; k++)
	{
		ITH[k+1]=ITH[k]+Nth+(k<Nextra);
	}
	
	//Create locks for the update of density/magnetization.
	vector<omp_lock_t> lock_sector(LX*LY);
	for (int k=0; k<LX*LY; k++)
	{
		omp_init_lock(&lock_sector[k]);
	}
	
	//Time evolution.
	for(int t=0;t<=tmax;t++)
	{
		//Exportations.
		if (t%500==0 or t==tmax)
		{
			const double rho=average(RHO,LX*LY), mx=average(MX,LX*LY), my=average(MY,LX*LY);
			const double mag=sqrt(mx*mx+my*my), phi=atan2(my,mx);
			fileAVERAGES << t << " " << rho << " " << mag << " " << phi << " " << mx << " " << my << endl;			
			cout << "time=" << t << " -rho=" << rho << " -mag=" << mag << " -phi=" << phi << running_time.TimeRun(" ") << endl;
			
			exportDynamics(RHO,MX,MY,beta,epsilon,rho0,q,LX,LY,init,RAN,t);
			exportParticles(beta,epsilon,rho0,q,LX,LY,init,RAN,t,Npart,PART);
		}
		if (t>teq)
		{
			AV.update(RHO,MX,MY,LX,LY);
		}
		if (AV.Nav>0 and (t%(tmax/100)==0 or t==tmax))
		{
			AV.exportFile(beta,epsilon,rho0,q,LX,LY,init,RAN);
		}
		
		//At each time step move Npart particles randomly choosen (multithreading).
		#pragma omp parallel
		{
			const int actual_thread=omp_get_thread_num();
			const int i0=ITH[actual_thread], N0=ITH[actual_thread+1]-i0;
			for (int i=0; i<N0; i++)
			{
				//The random particle is called j, with (local) position (X0,Y0) and spin SIGMA0.
				const int j=i0+int(N0*ran());
				const int X0=int(PART[j].x), Y0=int(PART[j].y), SIGMA0=PART[j].sigma, k0=X0*LY+Y0;
				
				//New orientation (uniformly chosen).
				int SIGMA=int((q-1)*ran());
				if (SIGMA>=SIGMA0)
				{
					SIGMA++;
				}
				if (SIGMA==SIGMA0)
				{
					cerr << "BAD VALUE OF SIGMA_NEW: " << SIGMA << " SIGMA0=" << SIGMA0 << endl;
					abort();
				}
				
				//Get the probability to flip throught the new orientation: DeltaH represents the sum of cos(phi-phik)-cos(PHI0-phik) with phik the orientation of neighbors, and rhoj the number of neighbors.
				//Get the index of neighbor boxes.
				const int xm=X0-1+(X0==0)*LX, xp=X0+1-(X0==LX-1)*LX;
				const int ym=Y0-1+(Y0==0)*LY, yp=Y0+1-(Y0==LY-1)*LY;
				int KN[9]={xm*LY + ym, xm*LY + Y0, xm*LY + yp, X0*LY + ym, X0*LY + Y0, X0*LY + yp, xp*LY + ym, xp*LY + Y0, xp*LY + yp};
				sort(KN, KN+9);
				
				//Lock neighbor boxes.
				for (int kn:KN)
				{
					omp_set_lock(&lock_sector[kn]);
				}				
				
				int rhoj=1;
				double DeltaH=0.;
				//Take the energy for particles in neighbour sectors with a distance smaller than 1.
				for (int kn:KN)
				{
					const vector<int> &neighbours=SEC.get(kn); //Particles in the sector kn.
					for (int k:neighbours)
					{
						if (j!=k and distance2(PART[j],PART[k],LX,LY)<1)
						{
							const int sigmak=PART[k].sigma;
							DeltaH+=COS_DIFF[SIGMA*q+sigmak]-COS_DIFF[SIGMA0*q+sigmak]; //Delta H = \sum_k [cos(phi-phik)-cos(phi0-phik)]
							rhoj++;
						}
					}
				}
				
				//Unlock neighbor boxes.
				for (int kn:KN)
				{
					omp_unset_lock(&lock_sector[kn]);
				}
				
				const double proba_flip=exp(beta*DeltaH/rhoj)*DeltaT;			
				
				//Verify that the probability to wait is positive.
				if (proba_hop+proba_flip>1)
				{
					cerr << "THE PROBABILITY TO WAIT IS NEGATIVE: proba_hop=" << proba_hop << " proba_flip=" << proba_flip << endl;
					cerr << "CHANGE THE VALUE OF DELTA_T !" << endl;
					abort();
				}
				
				const double random_number=ran();
				//The particle hops: perform the hopping on the particle and update the population of sectors.
				if (random_number<proba_hop)
				{
					//UPDATE OF SECTOR! Faster to directly update than to verify if the particle has effectively changed of sector before.
					PART[j].hop_guess(random_number/proba_hop,epsilon,q,LX,LY,COS,SIN);
					const int kN=int(PART[j].xnew)*LY+int(PART[j].ynew);
					
					if (kN==k0)
					{
						omp_set_lock(&lock_sector[k0]);
						PART[j].hop_apply();
						omp_unset_lock(&lock_sector[k0]);
					}
					else
					{
						const int kmin=min(k0,kN), kmax=max(k0,kN);
						
						omp_set_lock(&lock_sector[kmin]);
						omp_set_lock(&lock_sector[kmax]);
						
						SEC.remove(k0,j);
						RHO[k0]--;
						MX[k0]-=COS[SIGMA0];
						MY[k0]-=SIN[SIGMA0];
						
						PART[j].hop_apply();
						
						SEC.add(kN,j);
						RHO[kN]++;
						MX[kN]+=COS[SIGMA0];
						MY[kN]+=SIN[SIGMA0];
						
						omp_unset_lock(&lock_sector[kmax]);
						omp_unset_lock(&lock_sector[kmin]);
					}
				}
				//The particle flips: perform the flipping on the particle and modify the magnetizations (ind. of epsilon).
				else if (random_number<proba_hop+proba_flip)
				{
					omp_set_lock(&lock_sector[k0]);
					PART[j].flip(SIGMA);
					MX[k0]+=COS[SIGMA]-COS[SIGMA0];
					MY[k0]+=SIN[SIGMA]-SIN[SIGMA0];
					omp_unset_lock(&lock_sector[k0]);
					//NO UPDATE OF SECTORS! The particle has not moved.
				}
				//Else do nothing (proba_wait).		
			}
		}
		
	}
	
	//Destroy locks for the update of density/magnetization.
	for (int k=0; k<LX*LY; k++)
	{
		omp_destroy_lock(&lock_sector[k]);
	}

	return 0;
}
