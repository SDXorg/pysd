functions{
#include prey-predator_functions.stan

}
data{
    int <lower = 1> n_obs_state;
    //1. add data, 2 to n_obs_state, we need n_t (unlike draws2data confusingly..)
    int n_t;
    vector<lower=0>[2] y[n_t];  //measured stock
}

transformed data{
    real initial_time = 0.0;

    array[n_t] real times = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
}

parameters{
    real alpha;
    real gamma;
    real beta;
    real delta;

    // 2. add sigma
    real <lower = 0> sigma;
}

transformed parameters {
    # Initial ODE values
    real prey_initial = 30;
    real predator_initial = 4;

    // 3. change 2 to n_obs_state
    vector[2] initial_outcome;  # Initial ODE state vector
    initial_outcome[1] = prey_initial;
    initial_outcome[2] = predator_initial;

    vector[2] integrated_result[n_t] = ode_rk45(vensim_func, initial_outcome, initial_time, times, gamma, beta, delta, alpha);
}

model{
    alpha ~ normal(0.8, 0.1);
    gamma ~ normal(0.8, 0.1);
    beta ~ normal(0.05, 0.001);
    delta ~ normal(0.05, 0.001);
    
    // 4. added sampling dist error
    sigma ~ lognormal(-4.605170185988091, 1);

    // 5. likelihood statement from family 
    for (s in 1: n_obs_state){
        y[:, s] ~ lognormal(log(integrated_result[:, s]), sigma);
    }
}
//6. all new
generated quantities{
    array[n_t] vector[n_obs_state] y_hat;
    vector[n_t] log_lik;
    
    for (s in 1: n_obs_state){
        y_hat[:, s] = lognormal_rng(log(integrated_result[:, s]), sigma);
    } 
    for (s in 1: n_obs_state){
        //elementwise log likliehood
        log_lik[s] = lognormal_lpdf(y[s]|log(integrated_result[:, s]), sigma);
    }
}