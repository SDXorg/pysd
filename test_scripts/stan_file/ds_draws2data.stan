functions {
#include ds_relational.stan
}

data{
    int<lower =0> N;        // number of measurement times
    array[N] real times;         // measurement times
    real<lower = 0> customer_order_rate[N]; 
    real time_to_average_order_rate;
    real wip_adjustment_time;
    real manufacturing_cycle_time;
    real safety_stock_coverage;
    real inventory_coverage;

}

transformed parameters {
    real inventory_initial = 2 + 2 * 10000;
    real work_in_process_inventory_initial = 8 * fmax(0,10000 + 2 + 2 * 10000 - 2 + 2 * 10000 / 8);
    vector[2] initial_outcome = {inventory_initial, work_in_process_inventory_initial};

    vector<lower=0>[2] integrated_result_tilde[N] 
    = ode_rk45(vensim_func, initial_outcome, initial_time, times, 
               inventory_adjustment_time, minimum_order_processing_time,
               customer_order_rate);
}

generated quantities {
    vector<lower=0>[2] y_init_tilde; // simulated initial stock 
    vector<lower=0>[2] y_tilde[N];   // simulated stock 

    vector[2] sigma_tilde;
    vector[2] z_init_tilde;

    // U4 parameter uc
    real inventory_adjustment_time_tilde = 2; // abs(normal_rng(1, 0.5)); 
    real minimum_order_processing_time_tilde = 0.02; //abs(normal_rng(0.05, 0.05));   
    
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