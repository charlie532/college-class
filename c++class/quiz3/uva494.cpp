#include<iostream>
#include<string>

using namespace std;

int main(){
    string s;
    while( getline( cin, s ) ){
        int count = 0;
        bool inString = 0;
        for( int i = 0; i < s.length(); ++i ){
            if( isalpha(s[i]) && !inString ){
                inString = 1;
                count++;
            }
            else if( !isalpha(s[i]) && inString ){
                inString = 0;
            }
        }
        cout << count << endl;
    }
    return 0;
}