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
#include <sys/stat.h>

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
    if (!file.is_open()) {
        cerr << "Cannot open CSV file: " << filename << endl;
        exit(1);
    }
    string line;

    // Read header
    getline(file, line);
    vector<string> headers = split(line);
    int num_headers = (int)headers.size();

    // Map column name -> index
    unordered_map<string, int> colIndex;
    for (int i = 0; i < num_headers; i++) {
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
        if (line.empty()) continue;
        vector<string> tokens = split(line);

        // server_label may contain commas (e.g. "host:5201 (City, Country)")
        // which produces extra tokens; compute the offset dynamically
        int offset = (int)tokens.size() - num_headers;
        if (offset < 0) continue;

        Sample s;
        s.goodput  = stod(tokens[offset + colIndex["goodput_bps"]]);
        s.rtt      = stod(tokens[offset + colIndex["srtt_us"]]);
        s.loss     = stod(tokens[offset + colIndex["total_retrans"]]);
        s.cwnd     = stod(tokens[offset + colIndex["snd_cwnd_bytes"]]);

        string destination;
        for (int k = 0; k <= offset; k++) {
            if (k > 0) destination += ",";
            destination += tokens[k];
        }

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

string sanitize_filename(const string& server) {
    string filename = server;
    filename.erase(remove(filename.begin(), filename.end(), '\"'), filename.end());
    replace(filename.begin(), filename.end(), ' ', '_');
    replace(filename.begin(), filename.end(), '.', '_');
    replace(filename.begin(), filename.end(), ':', '_');
    return filename;
}

int main(int argc, char* argv[]) {
    string input_csv = "generated_data/tcp_stats.csv";
    string output_dir = "generated_data/predictions";

    if (argc > 1) input_csv = argv[1];
    if (argc > 2) output_dir = argv[2];

    mkdir(output_dir.c_str(), 0755);

    // Load Data
    map<string, vector<Sample>> data = loadCSV(input_csv);

    cout << "Loaded " << data.size() << " destinations from " << input_csv << endl;

    for(const auto& [server, raw_data] : data) {

        if ((int)raw_data.size() < WINDOW_SIZE + 2) {
            cerr << "Skipping " << server << ": too few samples (" << raw_data.size() << ")" << endl;
            continue;
        }

        // Normalization: convert cumulative loss to per-step delta
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

        if (X.empty()) continue;

        // Train-Test Split
        size_t split_idx = X.size() * TRAIN_RATIO;
        if (split_idx == 0 || split_idx >= X.size()) continue;

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

        // Output: full trace with predictions for test split
        string fname = sanitize_filename(server);
        string filepath = output_dir + "/" + fname + ".csv";
        ofstream plotFile(filepath);
        plotFile << "Time,Actual_CWND,Predicted_CWND" << endl;

        // Training phase: actual cwnd only (no prediction)
        for (size_t i = 0; i < split_idx; ++i) {
            plotFile << i << "," << trace[i + WINDOW_SIZE].cwnd << "," << endl;
        }

        // Test phase: actual cwnd + model prediction
        double current_sim_cwnd = trace[split_idx + WINDOW_SIZE].cwnd;
        for (size_t i = split_idx; i < X.size(); ++i) {
            double delta = model.predict(X[i]);
            current_sim_cwnd = max(1.0, current_sim_cwnd + delta);
            plotFile << i << "," << trace[i + WINDOW_SIZE].cwnd << "," << current_sim_cwnd << endl;
        }

        // Dump weights for reference
        string wpath = output_dir + "/" + fname + "_weights.txt";
        ofstream wfile(wpath);
        wfile << "weights:" << endl;
        for (size_t i = 0; i < model.weights.size(); ++i) {
            wfile << model.weights[i];
            if (i + 1 < model.weights.size()) wfile << ", ";
            if ((i + 1) % 3 == 0) wfile << endl;
        }
        wfile << "bias:" << endl << model.bias << endl;

        cout << "  " << server << " -> " << filepath << " (" << X.size() << " samples, split at " << split_idx << ")" << endl;
    }

    return 0;
}
