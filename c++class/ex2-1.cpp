#include <iostream>
#include <string.h>
#include <array>
using namespace std;

struct CStudent
{
   int id = 0,
       score = 0;
   char name[30] = {};
};

ostream & operator << (ostream &os, const CStudent &stu)
{
    os << stu.id << ' ' << stu.name << ' ' << stu.score;
    return os;
}

void sort(CStudent data[], int n){
    for (int i = n-1; i >= 1; i -= 1){
        for (int j = 0; j < i; j += 1){
            if (data[j].id > data[j+1].id){
                CStudent temp = data[j];
                data[j] = data[j+1];
                data[j+1] = temp;
            }
        }
    }
}


int main()
{
    CStudent data[] = {{1, 90, "David"}, {5, 100, "Allen"}, {3, 92, "Alice"}, {2, 85, "Bob"}, {4, 91, "Cyndi"}};
    const int Size = sizeof(data)/sizeof(data[0]);

    sort(data, Size);
    for (const auto &e: data) {cout << e << endl;} cout << endl;
}






