functions{
    #include prey-predator_functions.stan
}

data{
    int n_obs_state;
    int n_t;
    vector[20] predator_obs;
    vector[20] prey_obs;
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
    real sigma;
}

transformed parameters {
    // Initial ODE values
    real prey__init = 30;
    real predator__init = 4;

    vector[2] initial_outcome;  // Initial ODE state vector
    initial_outcome[1] = prey__init;
    initial_outcome[2] = predator__init;

    vector[2] integrated_result[n_t] = ode_rk45(vensim_ode_func, initial_outcome, initial_time, times, gamma, delta, alpha, beta);
    array[n_t] real prey = integrated_result[:, 1];
    array[n_t] real predator = integrated_result[:, 2];
}

model{
    alpha ~ normal(0.8, 0.1);
    gamma ~ normal(0.8, 0.1);
    beta ~ normal(0.05, 0.001);
    delta ~ normal(0.05, 0.001);
    sigma ~ lognormal(-4.605170185988091, 0.1);
    predator_obs ~ lognormal(predator, sigma);
    prey_obs ~ lognormal(prey, sigma);
}

