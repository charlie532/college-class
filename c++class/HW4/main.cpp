#include "String.cpp"
using namespace std;
int main() {
    String s("A");
    cout << s << endl;
    s += String("bc");
    cout << s << endl;
    s += "de";
    cout << s << endl;
    s += 'f';
    cout << s << endl;
    String s2(s);
    cout << "s:"<< s <<" s2:"<< s2 << endl;
    s2[1]='B';
    cout << "s:"<< s <<" s2:"<< s2 << endl;
    s[1]='C';
    cout << "s:"<< s <<" s2:"<< s2 << endl;
    s.swap(s2);
    cout << "s:"<< s <<" s2:"<< s2 << endl;
    char _cs[]="XYZ";
    char *cs = _cs;
    s.swap(cs);
    cout << "s:"<< s <<" s2:"<< s2 << endl;
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
    s=String("abc");
    s2='a';
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
    char _cs2[]="xyz";
    s2=_cs2;
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
    cin >> s;
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
    cout << "s:"<< (s+="12") <<" s2:"<< s2.c_str() << endl;
    cout << "s:"<< (s+='Y') <<" s2:"<< s2.c_str() << endl;
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
    s.append("hyu");
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
    s.append('d');
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
    s.append(String("bnt"));
    cout << "s:"<< s.c_str() <<" s2:"<< s2.c_str() << endl;
   
}   
 