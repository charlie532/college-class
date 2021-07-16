#include <iostream>
#include <regex>
#include <string>
using namespace std;

int main() {
    size_t T = 0; 
    cin >> T; 
    cin.ignore();
    for (size_t i=1; i<=T; i+=1) {
        string line;
        getline(cin, line);
        regex pat(R"(([A-Z])(\d+))");
        smatch sm;
        string ans;
        while (regex_search(line, sm, pat)){
            ans += string(stoi(sm[2].str()), sm[1].str()[0]);
            line = sm.suffix();
        }
        cout << "Case " << i << ": " << ans << endl;
    }
}