Rcpp::sourceCpp('/home/xiaolin/seminar/SDE/drift-final/drift-mu3.cpp')
set.seed(123)
library(yuima)
d=10
nu=20
fs<-function(k){
  z1=paste("b",c(1:d),sep="")
  z2=paste("x",c(1:d),sep="")
  z3=paste(z1,"+",z2,sep="")
  z22=z3[1]
  for(i in 2:d){
    z22=paste(z22,",",z3[i],sep="")
  }
  z22=paste("c(",z22,")",sep="")
  z4=paste("-(nu+d)*sum(",z22,"*","bm[,",k,"]",")",sep="")
  z5=paste("(nu+sum(",z22,"%*%","bm","%*%",z22,"))",sep="")
  zp=paste(z4,"/",z5,sep="")
  return(zp)
}
sol <-paste("x",c(1:d),sep="")
sol

b<-matrix("0",d,d)
diag(b)="2^0.5"

a<-c()
for(i in 1:d){
  a[i]=fs(i)
}
bb=rnorm(d)*(10^0.5)
bb=sort(bb)
ll<-list()
for(i in 1:d){
  z=paste("b",i,sep="")
  ll[[z]]=bb[i]
}
true.parameters <- ll
true.parameters

nd<-length(sol)
zz=rWishart(1,50,diag(1,nd,nd))
zz
eigen(zz[,,1])$value
bat=zz[,,1]
bat<-diag(1,d,d)
eigen(bat)$value
bm=solve(bat)
n <- 10^3
T=10
ysamp <- setSampling(Terminal = T, n = n)
ymodel <- setModel(drift = a, diffusion = b, solve.variable = sol)

#true.parameters <- list(mu1=2,mu2=1,mu3=0,mu4=1,mu5=2)
yuima <- setYuima(model = ymodel, sampling = ysamp)
yuima <- simulate(yuima,xinit=0,true.parameter = true.parameters)

plot(yuima@data)
true.parameters

x<-yuima@data
xx<-as.matrix(x@original.data)
dt=T/n
delxx<-xx[2:(n+1),]-xx[1:n,]
dim(delxx)
plot(delxx[,1],type="l")

library(coda)


z1<-fradp(N,g=0.5,xx,delxx,dt,bm)
z1[[1]]
res1[[4]]
N=3e4
res1=fr(N,g=0.5,xx,delxx,dt,bm)
#res1=fr(N,g=10,xx,delxx,dt)
accept(res1[[2]][(N-5e3):N,1:2])
#res0=fradp(N,g=0.01,g2=0.15,xx,delxx,yy,dt)
res21=fpcn(N,rho=0.95,xx,delxx,dt,ps=0,bm)
accept(res21[[2]][(N-2e3):N,1:2])
plot(res21[[1]][(N-2e4):N],type="l")

res31=fmpcn(N,rho=0.99,xx,delxx,dt,ps=0,bm)
accept(res31[[2]][(N-5e3):N,1:2])
plot(res31[[2]][(N-2e4):N,1],type="l")
res41=fgmpcn(N,rho=0.99,xx,delxx,dt,ps=0,bm)
accept(res41[[2]][(N-5e3):N,1:2])

res51=fmala(N,xx,delxx,dt,ep=1.0,bm)
accept(res51[[2]][(N-1e3):N,1:2])

N=3e3
res61=fpcn_margin(N,delta=7.5,xx,delxx,dt,ps=1,bm)
res81=fpcn_mala(N,delta=0.5,xx,delxx,dt,ps=1,bm)
accept(res81[[2]][(N-1e3):N,1:2])

res71=fhmc(N,xx,delxx,dt,L=1,ep=1.0,bm)
accept(res71[[2]][(N-1e3):N,1:2])

res91=fsplit_hmc(N,xx,delxx,dt,L=1,ep=0.2,ps=1,bm)
accept(res91[[2]][(N-1e3):N,1:2])


plot(res71[[1]][(N-1e3):N],type="l")
plot(res51[[1]][(N-3e3):N],type="l")
mean(res71[[1]][(N-3e3):N])
mean(res51[[1]][(N-3e3):N])

bn=2e4
ff<-function(res){
  N=length(res[[1]])
  nk=bn
  mm<-c()
  lik=res[[1]][(bn+1):N]
  k1<-effectiveSize(lik)
  t3=res[[3]]/1000
  mm[1]<-k1
  mm[2]<-t3
  mm[3]<-k1/t3
  mm[4]<-accept(res[[2]])
  return(mm)
}


kf<-function(z){
  dl=length(z)
  mm<-matrix(0,dl,4)
  colnames(mm)<-c("ESS","time-cost","ESS/second",
                  "acceptance_rate")
  pnames=c("RWM","PCN","MPCN","GMPCN","MALA","MGRAD","HMC",
           expression(paste(infinity,"-MALA",)),
           expression(paste(infinity,"-HMC")))
  rownames(mm)<-pnames[1:dl]
  for(i in 1:dl){
    res=get(paste("res",z[i],sep=""))
    mm[i,]<-ff(res)
  }
  return(round(mm,2))
}

kf(c(31,41,51))
ff(res31)
