functions {
    # Begin ODE declaration
    vector vensim_func(real time, vector outcome,     real alpha, real beta, real gamma, real delta    ){
        real prey = outcome[1];
        real predator = outcome[2];

        real prey_birth_rate = alpha * prey;
        real predator_death_rate = gamma * predator;
        real predator_birth_rate = delta * prey * predator;
        real prey_death_rate = beta * predator * prey;
        real prey_dydt = prey_birth_rate - prey_death_rate;
        real predator_dydt = predator_birth_rate - predator_death_rate;

        return {prey_dydt, predator_dydt};
    }
}

