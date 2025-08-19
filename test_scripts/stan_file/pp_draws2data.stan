functions {
#include pp_relational.stan
}

data{
    int<lower =0> N;        // number of measurement times
    array[N] real times;         // measurement times

}

generated quantities {
    vector<lower=0>[2] y_init_tilde; // simulated initial stock 
    vector<lower=0>[2] y_tilde[N];   // simulated stock 

    vector[2] sigma_tilde;
    vector[2] z_init_tilde;

    // U4 parameter uc
    real alpha_tilde = 0.55; // abs(normal_rng(1, 0.5)); 
    real beta_tilde = 0.028; //abs(normal_rng(0.05, 0.05)); 
    real gamma_tilde = 0.80; //abs(normal_rng(1, 0.5)); 
    real delta_tilde = 0.024; //abs(normal_rng(0.05, 0.05));    
    
    // U4 measurement uc
    z_init_tilde[1] = 30; //lognormal_rng(log(30), 1); 
    z_init_tilde[2] = 30; // lognormal_rng(log(30), 1); 
    
    // U4 different msr_err
    sigma_tilde[1] = 0.01; //lognormal_rng(-1, 1); 
    sigma_tilde[2] = 0.01; //lognormal_rng(-1, 1); 
    
    // calculate prior predictive
    vector<lower=0>[2] integrated_result_tilde[N]
        = ode_rk45(vensim_func, z_init_tilde, 0, times, 
                alpha_tilde, beta_tilde, gamma_tilde, delta_tilde);
    
    y_init_tilde = to_vector(lognormal_rng(log(z_init_tilde),
                                            sigma_tilde));
    
    for (n in 1:N) {
        //posterior predictive
        y_tilde[n] = to_vector(lognormal_rng(log(integrated_result_tilde[n]),
                                            sigma_tilde));
    }
}
