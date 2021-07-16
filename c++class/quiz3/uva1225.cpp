#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;

int main(){
    int T = 0;
    stringstream s;
    while(cin >> T){
        while( T-- ){
            int n = 0;
            int digit[10] = {0};
            vector<int> input;
            string ss;
            s.str("");
            s.clear();

            cin >> n;
            for( int i = 1; i <= n; ++i ){
                input.push_back(i);
            }
            for( int i = 0; i < input.size(); ++i ){
                s << input[i];
            }
            s >> ss;
            for( int i = 0; i < ss.size() ; ++i ){
                digit[ss[i] - '0']++;
            }
            for( int i = 0; i < 10; ++i ){
                cout << digit[i];
                if( i != 9 ){
                    cout << " ";
                }
            }
            cout << endl;
        }
    }
    return 0;
}