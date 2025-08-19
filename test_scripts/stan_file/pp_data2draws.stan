functions {
#include pp_relational.stan
}

data{
    int<lower =0> N;  // number of measurement times
    array[N] real times;   // measurement times
 
    vector<lower=0>[2] y_init; //init measured stock
    vector<lower=0>[2] y[N];  //measured stock
}

parameters{
  real<lower = 0> alpha; // est parameter 
  real<lower = 0> beta; // est parameter 
  real<lower = 0> gamma; // est paramete
  real<lower = 0> delta; // est parameter
  
  vector<lower = 0>[2] z_init;  // init state value
  vector<lower = 0>[2] sigma;   // msr error scale
}

transformed parameters {
     vector<lower=0>[2] integrated_result[N]
        = ode_rk45(vensim_func, z_init, 0, times, 
                    alpha, beta, gamma, delta);
}

model{
    // U4 parameter uc
    alpha ~ normal(.8, 0.1); // 1,1
    gamma ~ normal(.8, 0.1); // 1,1
    beta ~ normal(0.05, 0.01); // 0.05, 0.1
    gamma ~ normal(0.05, 0.01); // 0.05, 0.1
    
    // real alpha_tilde = 0.55; 
    // real beta_tilde = 0.028;
    // real gamma_tilde = 0.80;
    // real delta_tilde = 0.024;
    
    // U4 parameter uc
    sigma ~ lognormal(log(0.01), 1); //-1,1 
    
    // U4 parameter uc
    z_init ~ lognormal(log(100), 1); // E[log(z_init)] is `loc` of lognormal

    y_init ~ lognormal(log(z_init), sigma);
    
    for (n in 1:N) {
        y[n] ~ lognormal(log(integrated_result[n]), sigma);
    }
}

generated quantities {
    vector[N] log_lik;
    vector<lower=0>[2] y_hat[N];
    
    for(n in 1:N){
    //posterior predictive
    y_hat[n] = to_vector(lognormal_rng(log(integrated_result[n]), sigma));
    
    //elementwise log likliehood
    log_lik[n] = lognormal_lpdf(y[n]|log(integrated_result[n]), sigma);
    }
}