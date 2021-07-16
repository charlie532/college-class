#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
using namespace std;

int main(){
    map<string,string> transMap;
    vector<string> ans;
    string str;

    while( cin >> str && str != "#" ){
        string SortStr = str;
        transform( SortStr.begin(), SortStr.end(), SortStr.begin(), [](const char &s){ return tolower(s); } );
        sort( SortStr.begin(), SortStr.end() );
        if(transMap[SortStr].size()) transMap[SortStr]="#";
        else transMap[SortStr]=str;
    }
    for( map<string,string>::iterator i = transMap.begin(); i != transMap.end(); ++i ){
        if( i->second != "#" ){
            ans.push_back(i->second);
        }
    }
    sort( ans.begin(), ans.end() );
    for( int i = 0; i < ans.size(); ++i ){
        cout << ans[i] << endl;
    }
    return 0;
}