#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

int main(){
    string s, word;
    stringstream ss;
    bool isSpace;
    while( getline( cin, s ) ){
        ss.clear();
        ss.str(s);
        isSpace = false;
        while( ss >> word ){
            if( isSpace ) cout << " ";
            isSpace = true;
            reverse( word.begin(), word.begin() + word.length() );
            for( int i = 0; i < word.length(); ++i ){
                cout << word[i];
            }
        }
        cout << endl;
    }
    return 0;
}