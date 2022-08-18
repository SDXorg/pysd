functions {
    real table_for_order_fulfillment(real x){
        # x (0, 2) = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
        # y (0, 1) = (0.0, 0.2, 0.4, 0.58, 0.73, 0.85, 0.93, 0.97, 0.99, 1.0, 1.0)
        real slope;
        real intercept;

        if(x <= 0.2)
            intercept = 0.0;
            slope = (0.2 - 0.0) / (0.2 - 0.0);
            return intercept + slope * (x - 0.0);
        else if(x <= 0.4)
            intercept = 0.2;
            slope = (0.4 - 0.2) / (0.4 - 0.2);
            return intercept + slope * (x - 0.2);
        else if(x <= 0.6)
            intercept = 0.4;
            slope = (0.58 - 0.4) / (0.6 - 0.4);
            return intercept + slope * (x - 0.4);
        else if(x <= 0.8)
            intercept = 0.58;
            slope = (0.73 - 0.58) / (0.8 - 0.6);
            return intercept + slope * (x - 0.6);
        else if(x <= 1.0)
            intercept = 0.73;
            slope = (0.85 - 0.73) / (1.0 - 0.8);
            return intercept + slope * (x - 0.8);
        else if(x <= 1.2)
            intercept = 0.85;
            slope = (0.93 - 0.85) / (1.2 - 1.0);
            return intercept + slope * (x - 1.0);
        else if(x <= 1.4)
            intercept = 0.93;
            slope = (0.97 - 0.93) / (1.4 - 1.2);
            return intercept + slope * (x - 1.2);
        else if(x <= 1.6)
            intercept = 0.97;
            slope = (0.99 - 0.97) / (1.6 - 1.4);
            return intercept + slope * (x - 1.4);
        else if(x <= 1.8)
            intercept = 0.99;
            slope = (1.0 - 0.99) / (1.8 - 1.6);
            return intercept + slope * (x - 1.6);
        else if(x <= 2.0)
            intercept = 1.0;
            slope = (1.0 - 1.0) / (2.0 - 1.8);
            return intercept + slope * (x - 1.8);
    }

    # Begin ODE declaration
    vector vensim_func(real time, vector outcome,     real customer_order_rate    ){
        real inventory = outcome[1];
        real work_in_process_inventory = outcome[2];

        vector [2] dydt;
        real time_to_average_order_rate = 8;
        real change_in_exp_orders = customer_order_rate - expected_order_rate / time_to_average_order_rate;
        real expected_order_rate = change_in_exp_orders;
        real safety_stock_coverage = 2;
        real minimum_order_processing_time = 2;
        real desired_inventory_coverage = minimum_order_processing_time + safety_stock_coverage;
        real desired_inventory = desired_inventory_coverage * expected_order_rate;
        real inventory_adjustment_time = 8;
        real production_adjustment_from_inventory = desired_inventory - inventory / inventory_adjustment_time;
        real desired_production = fmax(0,expected_order_rate + production_adjustment_from_inventory);
        real manufacturing_cycle_time = 8;
        real desired_wip = manufacturing_cycle_time * desired_production;
        real wip_adjustment_time = 2;
        real adjustment_for_wip = desired_wip - work_in_process_inventory / wip_adjustment_time;
        real desired_production_start_rate = desired_production + adjustment_for_wip;
        real maximum_shipment_rate = inventory / minimum_order_processing_time;
        real desired_shipment_rate = customer_order_rate;
        real order_fulfillment_ratio = table_for_order_fulfillment(maximum_shipment_rate / desired_shipment_rate);
        real production_rate = work_in_process_inventory / manufacturing_cycle_time;
        real production_start_rate = fmax(0,desired_production_start_rate);
        real work_in_process_inventory_dydt = production_start_rate - production_rate;
        real shipment_rate = desired_shipment_rate * order_fulfillment_ratio;
        real inventory_dydt = production_rate - shipment_rate;
        dydt[1] = work_in_process_inventory_dydt;
        dydt[2] = inventory_dydt;
        
        return dydt;
    }
}
data{
    int<lower =0> N;        // number of measurement times
    array[N] real times;         // measurement times
    real<lower = 0> customer_order_rate[N]; 
}
transformed data{
}
parameters{
}
transformed parameters {
    real inventory_initial = 2 + 2 * 10000;
    real work_in_process_inventory_initial = 8 * fmax(0,10000 + 2 + 2 * 10000 - 2 + 2 * 10000 / 8);
    vector[2] initial_outcome = {inventory_initial, work_in_process_inventory_initial};
    vector<lower=0>[2] integrated_result = ode_rk45(vensim_func, initial_outcome, initial_time, times, customer_order_rate);
}
model{
}
generated quantities{
}
