functions {
#include ds_relational.stan
}

data{
    int<lower =0> N;        // number of measurement times
    array[N] real times;         // measurement times
    real<lower = 0> customer_order_rate[N]; 

}

transformed parameters {
    real inventory_initial = 2 + 2 * 10000;
    real work_in_process_inventory_initial = 8 * fmax(0,10000 + 2 + 2 * 10000 - 2 + 2 * 10000 / 8);
    vector[2] initial_outcome = {inventory_initial, work_in_process_inventory_initial};
    vector<lower=0>[2] integrated_result_tilde[N] = ode_rk45(vensim_func, initial_outcome, initial_time, times, customer_order_rate);
}
model{
}
generated quantities{
}
