// ------------------------
// C++ Version (My Custom Version - AC3 + Backtracking)
// ------------------------
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

const string digits = "123456789";
const string rows = "ABCDEFGHI";
const string cols = "123456789";
vector<string> squares;
unordered_map<string, vector<vector<string>>> units;
unordered_map<string, unordered_set<string>> peers;

vector<string> cross(const string& A, const string& B) {
    vector<string> result;
    for (char a : A)
        for (char b : B)
            result.push_back(string(1, a) + b);
    return result;
}

void initialize() {
    squares = cross(rows, cols);
    unordered_map<string, vector<vector<string>>> unit_map;
    unordered_map<string, unordered_set<string>> peer_map;

    vector<vector<string>> unitlist;
    for (char r : rows)
        unitlist.push_back(cross(string(1, r), cols));
    for (char c : cols)
        unitlist.push_back(cross(rows, string(1, c)));
    for (string rs : {"ABC", "DEF", "GHI"})
        for (string cs : {"123", "456", "789"})
            unitlist.push_back(cross(rs, cs));

    for (auto& s : squares) {
        vector<vector<string>> ulist;
        for (auto& u : unitlist)
            if (find(u.begin(), u.end(), s) != u.end())
                ulist.push_back(u);
        unit_map[s] = ulist;

        unordered_set<string> ps;
        for (auto& u : ulist)
            for (auto& sq : u)
                if (sq != s)
                    ps.insert(sq);
        peer_map[s] = ps;
    }
    units = unit_map;
    peers = peer_map;
}

bool eliminate(unordered_map<string, string>& values, string s, char d);
bool assign(unordered_map<string, string>& values, string s, char d);

bool eliminate(unordered_map<string, string>& values, string s, char d) {
    if (values[s].find(d) == string::npos) return true;
    values[s].erase(remove(values[s].begin(), values[s].end(), d), values[s].end());
    if (values[s].empty()) return false;
    if (values[s].size() == 1) {
        char d2 = values[s][0];
        for (string p : peers[s])
            if (!eliminate(values, p, d2))
                return false;
    }
    for (auto& u : units[s]) {
        vector<string> dplaces;
        for (string sq : u)
            if (values[sq].find(d) != string::npos)
                dplaces.push_back(sq);
        if (dplaces.empty()) return false;
        if (dplaces.size() == 1)
            if (!assign(values, dplaces[0], d))
                return false;
    }
    return true;
}

bool assign(unordered_map<string, string>& values, string s, char d) {
    string other_vals;
    for (char d2 : values[s])
        if (d2 != d) other_vals += d2;
    for (char d2 : other_vals)
        if (!eliminate(values, s, d2))
            return false;
    return true;
}

unordered_map<string, string> parse_grid(const string& grid) {
    unordered_map<string, string> values;
    for (string s : squares)
        values[s] = digits;
    for (int i = 0; i < grid.size(); ++i) {
        char d = grid[i];
        if (digits.find(d) != string::npos) {
            if (!assign(values, squares[i], d))
                return {};
        }
    }
    return values;
}

unordered_map<string, string> backtrack(unordered_map<string, string> values) {
    bool complete = true;
    for (auto& kv : values) {
        if (kv.second.size() != 1) {
            complete = false;
            break;
        }
    }
    if (complete) return values;

    string min_sq;
    int min_len = 10;
    for (auto& kv : values) {
        if (kv.second.size() > 1 && kv.second.size() < min_len) {
            min_sq = kv.first;
            min_len = kv.second.size();
        }
    }
    for (char d : values[min_sq]) {
        auto new_values = values;
        if (assign(new_values, min_sq, d)) {
            auto attempt = backtrack(new_values);
            if (!attempt.empty()) return attempt;
        }
    }
    return {};
}

string solve(const string& grid) {
    auto values = parse_grid(grid);
    if (values.empty()) return "Invalid";
    auto result = backtrack(values);
    if (result.empty()) return "Unsolvable";
    string output;
    for (string s : squares)
        output += result[s];
    return output;
}

int main() {
    initialize();
    ifstream infile("input.txt");
    ofstream outfile("output_cpp.txt");
    string line;
    vector<string> puzzles;
    while (getline(infile, line))
        puzzles.push_back(line);

    auto start = high_resolution_clock::now();
    for (const auto& puzzle : puzzles)
        outfile << solve(puzzle) << "\n";
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    cout << "C++ Solver Time: " << duration.count() << " ms\n";
    return 0;
}
