#include <iostream>
#include <string>
#include <regex>
using namespace std;

int main(){
    int T = 0;
    cin >> T; cin.ignore();
    while( T-- ){
        string first;
        string second; 
        getline(cin, first);
        getline(cin, second);
        regex pat("(.*)<(.*)>(.*)<(.*)>(.*)");
        smatch sm;
        regex_search(first, sm, pat);
        string out1;
        for (size_t i=1; i<sm.size(); i+=1){ 
            out1 += sm[i].str();
        }
        cout << out1 << endl;
        cout << second.substr(0, second.size()-3) << sm[4] << sm[3] << sm[2] << sm[5] << endl;
    }
}