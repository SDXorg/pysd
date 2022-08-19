functions {
#include ds_relational.stan
}

data{
    int<lower =0> N;        // number of measurement times
    array[N] real times;         // measurement times
    real<lower = 0> customer_order_rate[N];
    
    vector<lower=0>[2] y_init; //init measured stock
    vector<lower=0>[2] y[N];  //measured stock
}

parameters{
    real <lower = 6, upper =12> inventory_adjustment_time;
    real <lower = 1, upper =4> minimum_order_processing_time;
    
    vector<lower = 0>[2] z_init;  // init state value
    vector<lower = 0>[2] sigma;   // msr error scale
}

transformed parameters {
    real inventory_initial = 2 + 2 * 10000;
    real work_in_process_inventory_initial = 8 * fmax(0,10000 + 2 + 2 * 10000 - 2 + 2 * 10000 / 8);
    vector[2] initial_outcome = {inventory_initial, work_in_process_inventory_initial};
    vector<lower=0>[2] integrated_result 
        = ode_rk45(vensim_func, initial_outcome, initial_time, times, customer_order_rate);
}

model{
    // auto_prior from U4
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
        target += partial_sum_lpdf(log(integrated_result[n]), 1, N, 
    }
}

generated quantities{
}
