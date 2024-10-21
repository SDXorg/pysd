static int iteration_counter = 0;

// https://discourse.mc-stan.org/t/generating-random-numbers-in-the-model/3608
// https://discourse.mc-stan.org/t/is-it-possible-to-access-the-iteration-step-number-inside-a-stan-program/1871/6
// https://mc-stan.org/docs/cmdstan-guide/using-external-cpp-code.html
// https://discourse.mc-stan.org/t/hoping-for-some-guidance-help-with-implementing-custom-log-likelihood-and-gradient-for-research-project-details-below/24598/14
namespace vensim_ode_model_namespace {
    inline int get_current_iteration(std::ostream* pstream__) {
        return iteration_counter;
    }

    inline void increment_iteration(std::ostream* pstream__) {
        ++iteration_counter;
    }
}