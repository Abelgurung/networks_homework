#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

double calculate_ml_cwnd(double current_cwnd, 
                         const vector<double>& current_window, // The 15 features 
                         const vector<double>& learned_weights,
                         double learned_bias) {
    
    double predicted_delta = learned_bias;
    for (size_t i = 0; i < current_window.size(); ++i) {
        predicted_delta += learned_weights[i] * current_window[i];
    }

    // Assuming the last 3 elements in the window are the most recent (t-1) Goodput, RTT, and Loss
    double recent_loss = current_window[current_window.size() - 1]; 
    double recent_rtt = current_window[current_window.size() - 2];
    double recent_goodput = current_window[current_window.size() - 3];

    if (recent_loss > 0) {
        return current_cwnd * 0.5; // Multiplicative Decrease overrides the ML
    }
    
    double next_cwnd = current_cwnd + predicted_delta;

    // bind it to a minimum of 1 MSS (Maximum Segment Size, e.g., 1460 bytes).
    return max(1460.0, next_cwnd);
}

int main() {
    // Paste weights here (for 5 different traces):
    vector<double> weights = {
         -30862.603682, -6067.321151, 52987.586870, 
         31891.598018, 8315.862099, -2508.187687, 
         -9591.441400, 786.630767, -3908.453123, 
         15774.519791, 1670.597318, -2508.187687, 
         -12441.414770, 192.573020, -46252.140098
    };
    // Paste bias here
    double bias = 2780.402485;

    // double next = calculate_ml_cwnd(current_cwnd, X_test, weights, bias);

    // need to input: current_cwnd; X_test ( which contains 15 features: goodput, RTT, loss for time t-5, t-4, t-3, t-2, t-1)
}
