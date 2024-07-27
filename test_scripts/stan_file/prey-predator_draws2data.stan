functions{
#include prey-predator_functions.stan
}

data{
    int <lower = 1> n_obs_state;
}

transformed data{
    real initial_time = 0.0;
    int n_t = 20;
    array[n_t] real times = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
}

generated quantities{
    // 1.add real infront of all params + change from ~ to = , dist to dist_rng()
    real alpha_tilde = normal_rng(0.55, 0.1);
    real gamma_tilde  = normal_rng(0.8, 0.1);
    real beta_tilde  = normal_rng(0.028, 0.01);
    real delta_tilde = normal_rng(0.024, 0.01);

    // 2. manually moved from tp to gp 2. add _tilde to ode params in param ode_rk45() 
    // Initial ODE values
    real prey_initial = 30;
    real predator_initial = 4;

    // 3. 2 to n_obs_state
    vector[2] initial_outcome;  # Initial ODE state vector
    initial_outcome[1] = prey_initial;
    initial_outcome[2] = predator_initial;
    
    // 4. add tilde
    vector[2] integrated_result_tilde[n_t] = ode_rk45(vensim_func, initial_outcome, initial_time, times, gamma_tilde, beta_tilde, delta_tilde, alpha_tilde);
    
    //5. add sampling; sigma, y_tilde
    vector<lower=0>[2] y_tilde[n_t];  //measured stock
    real sigma_tilde = lognormal_rng(log(0.01), 0.01);
    
    //6. prior predictive

    for (s in 1: n_obs_state){
        y_tilde[:, s] = lognormal_rng(log(integrated_result_tilde[:, s]), sigma_tilde);
    } 
                                           
}