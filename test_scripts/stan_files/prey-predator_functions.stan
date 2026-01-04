// Begin ODE declaration
vector vensim_ode_func(real time, vector outcome, real gamma, real delta, real alpha, real beta){
    vector[2] dydt;  // Return vector of the ODE function

    // State variables
    real prey = outcome[1];
    real predator = outcome[2];

    real prey_death_rate = beta * predator * prey;
    real prey_birth_rate = alpha * prey;
    real prey_dydt = prey_birth_rate - prey_death_rate;
    real predator_death_rate = gamma * predator;
    real predator_birth_rate = delta * prey * predator;
    real predator_dydt = predator_birth_rate - predator_death_rate;

    dydt[1] = prey_dydt;
    dydt[2] = predator_dydt;

    return dydt;
}
