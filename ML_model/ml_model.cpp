#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <map>

using namespace std;

// Configuration
const int WINDOW_SIZE = 5;      // Model looks at 5 past time-steps
const double TRAIN_RATIO = 0.8; // 80% Training, 20% Testing
const double ALPHA = 1.0;       // RTT penalty
const double BETA = 50.0;      // Loss penalty

struct Sample {
    double goodput, rtt, loss, cwnd;
};

// Helper: split a string by comma
vector<string> split(const string& line) {
    vector<string> tokens;
    string token;
    stringstream ss(line);
    while (getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    return tokens;
}

map<string, vector<Sample>> loadCSV(const string& filename) {
    map<string, vector<Sample>> data;
    ifstream file(filename);
    string line;

    // Read header
    getline(file, line);
    vector<string> headers = split(line);

    // Map column name -> index
    unordered_map<string, int> colIndex;
    for (int i = 0; i < headers.size(); i++) {
        colIndex[headers[i]] = i;
    }

    // Check required fields exist
    if (colIndex.find("goodput_bps")  == colIndex.end() ||
        colIndex.find("srtt_us")      == colIndex.end() ||
        colIndex.find("total_retrans")     == colIndex.end() ||
        colIndex.find("snd_cwnd_bytes")     == colIndex.end()) {

        cerr << "Missing required columns in CSV!" << endl;
        exit(1);
    }

    // Read data lines
    while (getline(file, line)) {
        vector<string> tokens = split(line);

        Sample s;
        s.goodput  = stod(tokens[1 + colIndex["goodput_bps"]]);
        s.rtt      = stod(tokens[1 + colIndex["srtt_us"]]);
        s.loss     = stod(tokens[1 + colIndex["total_retrans"]]);
        s.cwnd     = stod(tokens[1 + colIndex["snd_cwnd_bytes"]]);
        string destination = tokens[0] + "," + tokens[1];

        data[destination].push_back(s);
    }

    return data;
}

vector<Sample> normalizeTrace(const vector<Sample>& input) {
    if (input.empty()) return {};

    int n = input.size();
    Sample mean = {0, 0, 0, 0};
    Sample stddev = {0, 0, 0, 0};

    // 1. Calculate Mean
    for (const auto& s : input) {
        mean.goodput += s.goodput;
        mean.rtt += s.rtt;
        mean.loss += s.loss;
    }
    mean.goodput /= n; mean.rtt /= n; mean.loss /= n;

    // 2. Calculate Standard Deviation
    for (const auto& s : input) {
        stddev.goodput += pow(s.goodput - mean.goodput, 2);
        stddev.rtt += pow(s.rtt - mean.rtt, 2);
        stddev.loss += pow(s.loss - mean.loss, 2);
    }
    stddev.goodput = sqrt(stddev.goodput / n) + 1e-9;
    stddev.rtt = sqrt(stddev.rtt / n) + 1e-9;
    stddev.loss = sqrt(stddev.loss / n) + 1e-9;

    // 3. Create Normalized Trace
    vector<Sample> normalized = input;
    for (auto& s : normalized) {
        s.goodput = (s.goodput - mean.goodput) / stddev.goodput;
        s.rtt = (s.rtt - mean.rtt) / stddev.rtt;
        s.loss = (s.loss - mean.loss) / stddev.loss;
    }

    return normalized;
}

// Objective Function
double eta(const Sample& s) {
    return s.goodput - (ALPHA * s.rtt) - (BETA * s.loss);
}

struct Model {
    vector<double> weights;
    double bias = 0.0;

    double predict(const vector<double>& x) {
        double res = bias;
        for(size_t i=0; i<x.size(); ++i) res += x[i] * weights[i];
        return res;
    }
};

int main() {
    // Load Data
    map<string, vector<Sample>> data = loadCSV("tcp_stats.csv");
    
    for(const auto& [server, raw_data] : data) {

        // Normalization
        vector<Sample> trace = raw_data;
        for(size_t i = 1; i < trace.size(); i++){
            trace[i].loss -= trace[i-1].loss;
        }
        trace = normalizeTrace(trace);
        
        // Build Windowed Dataset
        vector<vector<double>> X;
        vector<double> y;
        vector<double> rewards;

        for (size_t t = WINDOW_SIZE; t < trace.size(); ++t) {
            vector<double> history;
            for (int i = 1; i <= WINDOW_SIZE; ++i) {
                history.push_back(trace[t-i].goodput);
                history.push_back(trace[t-i].rtt);
                history.push_back(trace[t-i].loss);
            }
            X.push_back(history);
            y.push_back(trace[t].cwnd - trace[t-1].cwnd); // Action taken
            rewards.push_back(eta(trace[t])); // Utility function
        }

        // Train-Test Split
        size_t split_idx = X.size() * TRAIN_RATIO;
        
        // Training 
        Model model;
        model.weights.assign(X[0].size(), 0.0);
        double lr = 0.001;

        for (int epoch = 0; epoch < 5000; ++epoch) {
            for (size_t i = 0; i < split_idx; ++i) {
                // skip bad actions
                if (rewards[i] < 0) continue; 

                double pred = model.predict(X[i]);
                double error = pred - y[i];

                for (size_t j = 0; j < model.weights.size(); ++j) {
                    model.weights[j] -= lr * error * X[i][j];
                }
                model.bias -= lr * error;
            }
        }

        // Evaluation (Test Split)
        string filename = server;
        filename.erase(remove(filename.begin(), filename.end(), '\"'), filename.end());
        replace(filename.begin(), filename.end(), ' ', '_');
        replace(filename.begin(), filename.end(), '.', '_');
        replace(filename.begin(), filename.end(), ':', '_');
        filename = filename + ".csv";
        ofstream plotFile(filename);
        plotFile << "Time,Actual_CWND,Predicted_CWND" << endl;
        double current_sim_cwnd = trace[split_idx + WINDOW_SIZE].cwnd;
        
        for (size_t i = split_idx; i < X.size(); ++i) {
            double delta = model.predict(X[i]);
            current_sim_cwnd = max(1.0, current_sim_cwnd + delta);
            
            plotFile << i << "," << trace[i + WINDOW_SIZE].cwnd << "," << current_sim_cwnd << endl;
        }
    }

    return 0;
}