# Begin ODE declaration
vector vensim_func(real time, vector outcome, real gamma, real beta, real delta, real alpha){
    vector[2] dydt;  # Return vector of the ODE function

    # State variables
    real prey = outcome[1];
    real predator = outcome[2];

    real predator_birth_rate = delta * prey * predator;
    real predator_death_rate = gamma * predator;
    real prey_death_rate = beta * predator * prey;
    real predator_dydt = predator_birth_rate - predator_death_rate;
    real prey_birth_rate = alpha * prey;
    real prey_dydt = prey_birth_rate - prey_death_rate;

    dydt[1] = prey_dydt;
    dydt[2] = predator_dydt;

    return dydt;
}
