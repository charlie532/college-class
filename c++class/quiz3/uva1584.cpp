#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

int main() {
    string s;
    int n = 0;
    cin >> n;
    for( int i = 0; i < n; ++i ){
        cin >> s;
        string min = s;
        for( int j = 0; j < s.size(); ++j ){
            rotate( s.begin(), s.begin()+1, s.end());
            if( s < min ) min = s;
        }
        cout << min << endl;
    }
}