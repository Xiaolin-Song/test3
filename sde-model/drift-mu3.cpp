#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace arma;

static double const log2pi = std::log(2.0 * M_PI);

const double nu=20;
const double gt=0.5;
const int bn=1e3;

// set seed
// [[Rcpp::export]]
void set_seed(unsigned int seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);  
}


// [[Rcpp::export]]
arma::mat myfun(int d) {
  set_seed(123);
  arma::mat ma(d,d);
  for(int i=0;i<d;i++){
    for(int j=0;j<d;j++){
      ma(i,j)=10.0*std::pow(0.2,std::abs(i-j));
      //ma(i,j)=R::rnorm(0,1.0);
    }
  }
  return(ma);
}



// [[Rcpp::export]]
double dmvnrm(arma::rowvec x,  
              arma::rowvec mean,  
              arma::mat sigma){  
  int xdim = x.n_cols;
  arma::mat invsigma=arma::inv(sigma);
  arma::mat const rooti = arma::inv(trimatu(arma::chol(sigma)));
  double const rootisum = arma::sum(log(rooti.diag())), 
    constants = -(double)xdim/2.0 * log2pi, other_terms = rootisum + constants;
  double out=other_terms - 0.5 * arma::sum(x*invsigma*x.t());
  //out=exp(out);
  return(out) ;
}


/* accept rate*/

// [[Rcpp::export]]
double accept(arma::mat x){
  int nn=x.n_rows;
  int s=0;
  for(int i=1;i<nn;i++){
    //arma::umat ss=(x.row(i)==x.row(i-1));
    if(x(i,0)==x(i-1,0)){
      s=s;
    }else{
      s=s+1;
    }
  }
  double ac=s/(double)nn;
  return(ac);
}


//loglikelihood

// [[Rcpp::export]]
double loglik(arma::rowvec alpha,arma::mat xx,arma::mat delxx,double dt,arma::mat insig){
  double sd=std::pow(10,0.5);
  double prior=-0.5*arma::sum(alpha%alpha)/(sd*sd);
  int nn=delxx.col(0).n_elem; //observation number
  int n=delxx.row(0).n_elem;  //parameeter dimension
  
  arma::mat I(n,n);I=1.0/(double)2*I.eye();
  double loglik=0;
  double K=(n+nu);
  for(int i=0;i<nn;i++){
    arma::rowvec pv=xx.row(i)+alpha;
    arma::rowvec bi=-K*(pv*insig)/(nu+arma::sum(pv*insig*pv.t()));
    arma::rowvec vx=delxx.row(i)-dt*bi;
    loglik=-0.5*(1.0/(dt)*arma::sum(vx*I*vx.t()))+loglik;
  }
  return(loglik+prior);
}


//pre-condition on /mu
// [[Rcpp::export]]
arma::vec loglikm(arma::rowvec alpha,arma::mat xx,arma::mat delxx,double dt,arma::rowvec va,arma::mat insig){
  arma::vec zl(2);
  
  double b=std::pow(10,0.5);
  
  
  double ps=-0.5*arma::sum((alpha%alpha)/(b*b));
  
  alpha=alpha+va;
  double prior=-0.5*arma::sum((alpha%alpha)/(b*b));
  
  int nn=delxx.col(0).n_elem; //observation number
  int n=delxx.row(0).n_elem; //parameter dimension

    arma::mat I(n,n);I=1.0/(double)2*I.eye();
  double loglik=0;
  double K=(n+nu);
  for(int i=0;i<nn;i++){
    arma::rowvec pv=xx.row(i)+alpha;//alpha.subvec(5,9);
    arma::rowvec bi=-K*(pv*insig)/(nu+arma::sum(pv*insig*pv.t()));
    arma::rowvec vx=delxx.row(i)-dt*bi;
    loglik=-0.5*(1.0/(dt)*arma::sum(vx*I*vx.t()))+loglik;
  }
  
  zl(0)=loglik+prior-ps;
  zl(1)=loglik+prior;
  return(zl);
}



// [[Rcpp::export]]
arma::rowvec dif(arma::rowvec alpha,arma::mat xx,arma::mat delxx,double dt,arma::mat insig){
  int nn=delxx.col(0).n_elem;
  int n=delxx.row(0).n_elem;
  double b=std::pow(10,0.5);
  arma::rowvec prior(n);//=-alpha*1.0/(double)50;
  for(int i=0;i<n;i++){
    prior(i)=-alpha(i)/(b*b);
  }
  arma::rowvec vv(n);vv=arma::linspace<rowvec>(1,n,n);
  arma::mat I(n,n);I=1.0/(double)2*I.eye();
  double K=(n+nu);
  arma::rowvec dif(n);dif=prior; 
  for(int i=0;i<nn;i++){
    arma::rowvec pv=xx.row(i)+alpha;//alpha.subvec(5,9);
    arma::rowvec bi=-K*(pv*insig)/(nu+arma::sum(pv*insig*pv.t()));
    arma::rowvec vx=delxx.row(i)-dt*bi;
    arma::mat B(n,n);
    double aa=(nu+arma::sum(pv*insig*pv.t()));
    arma::rowvec x=xx.row(i);
    
    for(int j=0;j<n;j++){
      for(int k=0;k<n;k++){
        B(j,k)=1.0/(aa*aa)*(aa*insig(j,k)-arma::sum((x+alpha)*(insig.col(k)))*arma::sum((2.0*alpha+2.0*x)*insig.col(j)));
      }
    }
    
    dif=-0.5*K*vx*B.t()+dif;
  }
  
  return(dif);
}
// 
// 
/* rwm*/
// [[Rcpp::export]]
List fr(int N, double g,arma::mat xx,arma::mat delxx,double dt,arma::mat insig){
  List out(4);
  
  //double lower=0.01,upper=5.0;
  double b=std::pow(10,0.5);
  
  arma::rowvec beta_fix(3);beta_fix.ones();
  
  int n=delxx.n_cols;
  arma::mat C(n,n);C.eye();
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  arma::rowvec rd(n),pro(n),log_like(N),zz(n);
  zz=res.row(0)*b;
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  clock_t start=std::clock();
  
  
  for(int i=1;i<N;i++){
    rd=rd.randn();
    pro=res.row(i-1)+2.38*rd/(std::pow(n,0.5))*C*g;
    // if(sum(pro<lower)+sum(pro>upper)==0){
    //   break;
    // }
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    zz=zz+res.row(i);
  }
  zz=zz/(1.0*N);
  
  
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  //arma::mat ress = arma::join_rows(res,resa);
  //arma::mat lik = arma::join_cols(log_like,log_likea);
  out(0)=log_like;out(1)=res;out(2)=t4;out(3)=zz;
  return(out);
}


/* rwm*/
// [[Rcpp::export]]
List fradp(int N, double g,arma::mat xx,arma::mat delxx,double dt,arma::mat insig){
  List out(4);
  double b=std::pow(10,0.5);

  //double lower=0.01,upper=5.0;

  arma::rowvec beta_fix(3);beta_fix.ones();

  int n=delxx.n_cols;
  arma::mat C(n,n);C.eye();
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  arma::rowvec rd(n),pro(n),log_like(N),zz(n);
  res.row(0).randn();
  zz=res.row(0)*b;
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);

  clock_t start=std::clock();


  for(int i=1;i<N;i++){
    rd=rd.randn();
    pro=res.row(i-1)+2.38*rd/(std::pow(n,0.5))*C*g;
    // if(sum(pro<lower)+sum(pro>upper)==0){
    //   break;
    // }
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
    if(i>0.5*N){zz=zz+res.row(i);}
  }


  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  //arma::mat ress = arma::join_rows(res,resa);
  //arma::mat lik = arma::join_cols(log_like,log_likea);
  zz=zz/(0.5*N);
  out(0)=zz;out(1)=res;out(2)=t4;
  return(out);
}


//pcn
// [[Rcpp::export]]
List fpcn(int N, double rho,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig){
  List out(4);
  
  double lower=0.01,upper=5.0;
  
  arma::rowvec beta_fix(3);beta_fix.ones();
  int n=delxx.n_cols;
  
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat C(n,n);C=C.eye();
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  res.row(0)=sd*res.row(0);
  
  arma::rowvec vv(n);
  arma::rowvec rd(n),pro(n),log_like(N),zz(n);
  int stt=1;
  if(ps==1){//with pre-condition on /mu;
    List ls=fr(bn,gt,xx,delxx,dt,insig);
    arma::rowvec v1=ls(3);
    arma::mat burn=ls(1);
    arma::rowvec lik=ls(0);
    vv=v1;
    res.rows(0,bn-1)=burn;
    log_like.subvec(0,bn-1)=lik;
    stt=bn;
  }else{
    vv.zeros();
    log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  }
  
  clock_t start=std::clock();
  
  
  for(int i=stt;i<N;i++){
    rd=rd.randn()*sd;
    pro=vv+std::pow(rho,0.5)*(res.row(i-1)-vv)+std::pow(1.0-rho,0.5)*rd;
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1)+0.5*arma::sum((pro-vv)%(pro-vv))/sdd-0.5*arma::sum((res.row(i-1)-vv)%(res.row(i-1)-vv))/sdd;
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  
  clock_t end=std::clock();
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);

  out(0)=log_like;out(1)=res;out(2)=t4;
  return(out);
}



//mpcn
// [[Rcpp::export]]
List fmpcn(int N, double rho,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig){
  List out(4);
  double lower=0.01,upper=5.0;
  
  int n=delxx.n_cols;
  
  arma::rowvec vv(n);
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat C(n,n);C=C.eye();
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  res.row(0)=sd*res.row(0);
  
  arma::rowvec rd(n),pro(n),log_like(N),zz(n);
  int stt=1;
  if(ps==1){//with pre-condition on /mu;
    List ls=fr(bn,gt,xx,delxx,dt,insig);
    arma::rowvec v1=ls(3);
    arma::mat burn=ls(1);
    arma::rowvec lik=ls(0);
    vv=v1;
    res.rows(0,bn-1)=burn;
    log_like.subvec(0,bn-1)=lik;
    stt=bn;
  }else{
    vv.zeros();
    zz=res.row(0)*sd;
    log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  }
  
  clock_t start=std::clock();
  
  double a;
  
  double aa=arma::norm((res.row(stt-1)-vv)/sd,2),bb;
  for(int i=stt;i<N;i++){
    double gg=R::rgamma(0.5*n,2.0/(aa*aa));
    rd=rd.randn()*sd;
    pro=vv+std::pow(rho,0.5)*(res.row(i-1)-vv)+std::pow(1.0-rho,0.5)*rd*std::pow(gg,-0.5);
    
    double bb=arma::norm((pro-vv)/sd,2);
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1)+n*std::log(bb)-n*std::log(aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  clock_t end=std::clock();
  
  double t3 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  out(0)=log_like;out(1)=res;out(2)=t3;
  return(out);
}


/* rwm*/
// [[Rcpp::export]]
List fgmpcn(int N, double rho,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig){
  List out(4);
  
  double lower=0.01,upper=5.0;
  int n=delxx.n_cols;
  arma::rowvec beta_fix(3);beta_fix.ones();
  
  arma::rowvec vv(n);
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat C(n,n);C=C.eye();
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  res.row(0)=sd*res.row(0);
  
  arma::rowvec rd(n),pro(n),log_like(N),zz(n),apro(n);
  int stt=1;
  if(ps==1){//with pre-condition on /mu;
    List ls=fr(bn,gt,xx,delxx,dt,insig);
    arma::rowvec v1=ls(3);
    arma::mat burn=ls(1);
    arma::rowvec lik=ls(0);
    vv=v1;
    res.rows(0,bn-1)=burn;
    log_like.subvec(0,bn-1)=lik;
    stt=bn;
  }else{
    vv.zeros();
    zz=res.row(0)*sd;
    log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  }
  
  
  //double sd=std::pow(100,0.5),sdd=sd*sd;

  arma::mat sig=C*sdd;
  C=arma::chol(sig);//arma::rowvec vv=ls(2);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::rowvec orgv=(res.row(stt-1)-vv)*invsnew;
  
  clock_t start=std::clock();
  

  double a;int v=1;
  
  double aa=arma::norm(orgv,2),bb;
  for(int i=stt;i<N;i++){
    double bb;
    if(v==1){
      for(;;){
        double gg=R::rgamma(0.5*n,2.0/(aa*aa));
        rd=rd.randn();
        apro=std::pow(rho,0.5)*orgv+std::pow(1.0-rho,0.5)*rd*std::pow(gg,-0.5);
        bb=arma::norm(apro,2);
        if(bb>aa){
          break;
        }
      }
    }else{
      for(;;){
        double gg=R::rgamma(0.5*n,2.0/(aa*aa));
        rd=rd.randn();
        apro=std::pow(rho,0.5)*orgv+std::pow(1.0-rho,0.5)*rd*std::pow(gg,-0.5);
        bb=arma::norm(apro,2);
        if(bb<aa){
          break;
        }
      }
    }
    pro=vv+apro*C;
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1)+n*std::log(bb)-n*std::log(aa);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      orgv=apro;
      aa=bb;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      v=-v;
    }
    zz=zz+res.row(i);
  }
  
  
  clock_t end=std::clock();
  
  arma::rowvec alpha_fix=zz/(double)N;
  //
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  
  out(0)=log_like;out(1)=res;out(2)=t4,out(3)=vv;
  return(out);
}


/* mala */
// [[Rcpp::export]]
List fmala(int N,arma::mat xx,arma::mat delxx,double dt,double ep,arma::mat insig){
  
  int n=delxx.n_cols;
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat sig(n,n);sig=sig.eye()*sdd;
  arma::mat ssig=sig;
  List out(5);
  uvec IDX = regspace<uvec>(0,25,n-1);
  arma::mat C(n,n);C=arma::chol(sig);
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.zeros();
  arma::rowvec rd(n),log_like(N),pro(n);
  
  res.row(0)=0.1*rd.randn();//arma::randn(n).t();
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  clock_t start=std::clock();
  arma::rowvec gx=dif(res.row(0),xx,delxx,dt,insig);
  for(int i=1;i<N;i++){
    rd=rd.randn()*sd;
    pro=res.row(i-1)+0.5*ep*ep/sdd*gx+ep/sdd*rd;
    arma::rowvec gy=dif(pro,xx,delxx,dt,insig);
    arma::rowvec gp=(res.row(i-1)-pro-0.5*ep*ep/sdd*gy)*sdd/ep;
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-0.5*arma::sum(gp*gp.t())/sdd-(log_like(i-1)-0.5*arma::sum(rd*rd.t())/sdd);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      gx=gy;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
    }
  }
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  out(0)=log_like;out(1)=res;out(2)=t4;out(3)=ep;
  return(out);
}


// [[Rcpp::export]]
arma::rowvec dsign(arma::rowvec x){
  int n=x.n_elem;
  arma::rowvec zx=x;
  for(int i=0;i<n;i++){
    if(x(i)>0){
      zx(i)=1;
    }else{
      zx(i)=-1;
    }
  }
  return(zx);
}

/* pcn */
// [[Rcpp::export]]
List fpcn_margin(int N, double delta,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig){
  
  List out(5);
  int n=delxx.n_cols;
  arma::rowvec vv(n);
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat C(n,n);C=C.eye();
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();
  res.row(0)=sd*res.row(0);
  
  arma::rowvec rd(n),pro(n),log_like(N),zz(n),apro(n);
  int stt=1;
  if(ps==1){//with pre-condition on /mu;
    List ls=fr(bn,gt,xx,delxx,dt,insig);
    arma::rowvec v1=ls(3);
    arma::mat burn=ls(1);
    arma::rowvec lik=ls(0);
    vv=v1;
    res.rows(0,bn-1)=burn;
    log_like.subvec(0,bn-1)=lik;
    stt=bn;
  }else{
    vv.zeros();
    zz=res.row(0)*sd;
    log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  }
  arma::rowvec log_likef(N);
  
  arma::mat I(n,n);
  arma::mat sig=I.eye()*sdd;
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  //
  arma::mat A=delta/2.0*arma::inv_sympd(sig+0.5*delta*I)*sig;
  arma::mat pm=(2.0/delta*A*A.t()+A);
  arma::mat iva=arma::inv(2.0/delta*A+I);
  arma::mat cha=arma::chol(pm);
  
  arma::mat copm=res;
  int sk=stt-1;
  copm.row(sk)=res.row(sk)-vv;
  arma::vec zl=loglikm(copm.row(sk),xx,delxx,dt,vv,insig);
  log_likef(sk)=zl(0);
  
  //rho=0.4;//////######
  
  clock_t start=std::clock();
  //
  arma::rowvec gx(n),gy(n),zp(n),zpro(n),ds(n);
  gx=dif(copm.row(sk)+vv,xx,delxx,dt,insig)+(copm.row(sk))*invsig;
  for(int i=stt;i<N;i++){
    zpro=copm.row(i-1)+0.5*delta*gx;
    pro=2.0/delta*(zpro)*A+rd.randn()*cha;
    arma::vec aa=loglikm(pro,xx,delxx,dt,vv,insig);
    gy=dif(pro+vv,xx,delxx,dt,insig)+pro*invsig;
    
    double a=aa(0);
    double acc=a+arma::sum((copm.row(i-1)-2.0/delta*(pro+delta/4.0*gy)*A)*iva*gy.t())-
      log_likef(i-1)-arma::sum((pro-2.0/delta*(copm.row(i-1)+delta/4.0*gx)*A)*iva*gx.t());
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      copm.row(i)=pro;
      log_likef(i)=a;
      log_like(i)=aa(1);
      gx=gy;
      res.row(i)=vv+pro;
    }else{
      copm.row(i)=copm.row(i-1);
      log_likef(i)=log_likef(i-1);
      log_like(i)=log_like(i-1);
      res.row(i)=res.row(i-1);
    }
  }
  
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  
  out(0)=log_like;out(1)=res;out(2)=t4,out(3)=vv;
  return(out);
}




/*hmc*/
// [[Rcpp::export]]
List fhmc(int N,arma::mat xx,arma::mat delxx,double dt,int L,double ep,arma::mat insig){
  List out(4);
  int n=delxx.n_cols;
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat sig(n,n);sig=sig.eye()*sdd;
  arma::mat ssig=sig;
  
  uvec IDX = regspace<uvec>(0,25,n-1);
  arma::mat C(n,n);C=arma::chol(sig);
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.zeros();
  
  arma::rowvec rd(n),rw(n),pro(n),log_like(N),v(N);
  log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  
  clock_t start=std::clock();
  
  for(int i=1;i<N;i++){
    arma::mat zm(L+1,n),zv(L+1,n);
    zm.zeros();
    zv=zv.randn()*sd;
    zm.row(0)=res.row(i-1);
    // for(int j=1;j<(L+1);j++){
    //   zv.row(j)=zv.row(j-1)+0.5*ep*dif(zm.row(j-1),xx,delxx,dt,insig);
    //   zm.row(j)=zm.row(j-1)+ep*zv.row(j)/(sd*sd);
    //   zv.row(j)=zv.row(j)+0.5*ep*dif(zm.row(j),xx,delxx,dt,insig);
    // }
    // pro=zm.row(L);
    
    zv.row(1)=zv.row(0)+0.5*ep*dif(zm.row(0),xx,delxx,dt,insig);
    for(int j=1;j<(L+1);j++){
      zm.row(j)=zm.row(j-1)+ep*zv.row(j)/(sd*sd);
      if(j<L){
        zv.row(j+1)=zv.row(j)+ep*dif(zm.row(j),xx,delxx,dt,insig);
      }
    }
    zv.row(L)=zv.row(L)+0.5*ep*dif(zm.row(L),xx,delxx,dt,insig);
    pro=zm.row(L);
    double a=loglik(pro,xx,delxx,dt,insig);
    double acc=a-log_like(i-1)-0.5*arma::as_scalar(zv.row(L)*(zv.row(L)).t())/(sd*sd)+
      0.5*arma::as_scalar(zv.row(0)*zv.row(0).t())/(sd*sd);
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro;
      log_like(i)=a;
      //v(i)=v(i-1);
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      //v(i)=-1*v(i-1);
    }
    
  }
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  arma::mat ress=res.cols(IDX);
  out(0)=log_like;out(1)=res;out(2)=t4;
  return(out);
}



/* pcn */
// [[Rcpp::export]]
List fpcn_mala(int N, double delta,arma::mat xx,arma::mat delxx,double dt,int ps,arma::mat insig){
  
  List out(5);
  int n=delxx.n_cols;
  
  arma::mat res(N,n);res.zeros();
  res.row(0).randn();//arma::randn(n).t();
  arma::rowvec rd(n),log_like(N),pro(n),log_likef(N);
  
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat I(n,n);
  arma::mat sig=I.eye()*sdd;
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat clp=arma::chol(sig);
  arma::mat prim(n,n);prim=sig;
  
  res.row(0).randn();//arma::randn(n).t();
  arma::rowvec vv(n);
  arma::rowvec zz(n),apro(n);
  int stt=1;
  if(ps==1){//with pre-condition on /mu;
    List ls=fr(bn,gt,xx,delxx,dt,insig);
    arma::rowvec v1=ls(3);
    arma::mat burn=ls(1);
    arma::rowvec lik=ls(0);
    vv=v1;
    res.rows(0,bn-1)=burn;
    log_like.subvec(0,bn-1)=lik;
    stt=bn;
  }else{
    vv.zeros();
    zz=res.row(0)*sd;
    log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  }

  int sk=stt-1;
  
  //rho=0.4;//////######
  clock_t start=std::clock();
  
  
  arma::mat copm=res;
  copm.row(sk)=res.row(sk)-vv;
  arma::vec zl=loglikm(copm.row(sk),xx,delxx,dt,vv,insig);
  log_likef(sk)=zl(0);
  
  arma::rowvec gx(n),gy(n),zp(n),zpro(n);
  gx=dif(copm.row(sk)+vv,xx,delxx,dt,insig)+(copm.row(sk))*invsig;
  double r1=(2.0-delta)/(2.0+delta),r2=(2.0*delta)/(2.0+delta),r3=std::pow(8.0*delta,0.5)/(2.0+delta);
  
  for(int i=stt;i<N;i++){
    rd=rd.randn()*clp;
    pro=r1*(copm.row(i-1))+r2*gx*prim+r3*rd;
    arma::vec aa=loglikm(pro,xx,delxx,dt,vv,insig);
    gy=dif(pro+vv,xx,delxx,dt,insig)+(pro)*invsig;
    double a=aa(0);
    double acc=a+0.5*arma::sum((copm.row(i-1)-pro)%gy)+delta/4.0*arma::sum((copm.row(i-1)+pro)%gy)-
      delta/4.0*arma::sum(gy*sig*gy.t())
      -log_likef(i-1)-(0.5*arma::sum((pro-copm.row(i-1))%gx)+delta/4.0*arma::sum((copm.row(i-1)+pro)%gx)-
        delta/4.0*arma::sum(gx*sig*gx.t()));
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      copm.row(i)=pro;
      log_likef(i)=a;
      log_like(i)=aa(1);
      res.row(i)=pro+vv;
      gx=gy;
    }else{
      copm.row(i)=copm.row(i-1);
      log_likef(i)=log_likef(i-1);
      log_like(i)=log_like(i-1);
      res.row(i)=res.row(i-1);
    }
  }
  
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  out(0)=log_like;out(1)=res;out(2)=t4;
  return(out);
}



// [[Rcpp::export]]
List fsplit_hmc(int N,arma::mat xx,arma::mat delxx,double dt,int L,double ep,int ps,arma::mat insig){
  
  List out(4);
  int n=delxx.n_cols;
  double sd=std::pow(10,0.5),sdd=sd*sd;
  arma::mat sig(n,n);sig=sig.eye()*sdd;
  arma::mat ssig=sig;
  
  arma::mat C(n,n);C=arma::chol(sig);
  arma::mat invsig=arma::inv(sig);
  arma:: mat invsnew=arma::chol(arma::inv(sig)).t();
  arma::mat res(N,n);res.zeros();
  
  arma::mat clp=arma::chol(sig);
  arma::mat prim(n,n);prim=sig;
  
  
  res.row(0).randn();//arma::randn(n).t();
  arma::rowvec rd(n),log_like(N),pro(n),log_likef(N),vv(n);
  int stt=1;
  if(ps==1){//with pre-condition on /mu;
    List ls=fr(bn,gt,xx,delxx,dt,insig);
    arma::rowvec v1=ls(3);
    arma::mat burn=ls(1);
    arma::rowvec lik=ls(0);
    vv=v1;
    res.rows(0,bn-1)=burn;
    log_like.subvec(0,bn-1)=lik;
    stt=bn;
  }else{
    vv.zeros();
    log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  }
  
  int sk=stt-1;
  
  
  //rho=0.4;//////######
  clock_t start=std::clock();
  
  //log_like(0)=loglik(res.row(0),xx,delxx,dt,insig);
  
  arma::rowvec gx(n),gy(n),zp(n),zpro(n),zm_new(n),zv_new(n);
  arma::mat zm(L+1,n),zv(L+1,n);
  arma::mat comp=res;
  comp.row(sk)=res.row(sk)-vv;
  
  for(int i=stt;i<N;i++){
    zm.zeros();
    zv=zv.randn()*clp;
    zm.row(0)=comp.row(i-1);
    //zv.row(1)=zv.row(0)-0.5*ep*dif(zm.row(0));
    for(int j=1;j<(L+1);j++){
      zm_new=std::cos(0.5*ep)*zm.row(j-1)+std::sin(ep*0.5)*zv.row(j-1);
      zv_new=-std::sin(0.5*ep)*zm.row(j-1)+std::cos(ep*0.5)*zv.row(j-1);
      
      zv_new=zv_new+ep*(dif(zm_new+vv,xx,delxx,dt,insig)+(zm_new)*invsig)*prim;
      
      zm.row(j)=std::cos(0.5*ep)*zm_new+std::sin(ep*0.5)*zv_new;
      zv.row(j)=-std::sin(0.5*ep)*zm_new+std::cos(ep*0.5)*zv_new;
    }
    
    pro=zm.row(L);
    double a=loglik(pro+vv,xx,delxx,dt,insig);
    double acc=a-log_like(i-1)-0.5*arma::as_scalar((zv.row(L))*invsig*(zv.row(L)).t())+
      0.5*arma::as_scalar((zv.row(0))*invsig*(zv.row(0)).t());
    double u=std::log(arma::as_scalar(arma::randu(1)));
    if(acc>u){
      res.row(i)=pro+vv;
      log_like(i)=a;
      comp.row(i)=pro;
    }else{
      res.row(i)=res.row(i-1);
      log_like(i)=log_like(i-1);
      comp.row(i)=comp.row(i-1);
    }
  }
  double t4 = (std::clock() - start )/(CLOCKS_PER_SEC / 1000);
  
  out(0)=log_like;out(1)=res;out(2)=t4;
  return(out);
}
