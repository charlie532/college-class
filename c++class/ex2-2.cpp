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
using functionPointer = int(*)(const CStudent &lhs, const CStudent &rhs);
void sort(CStudent data[], int n, functionPointer cmp){
    for (int i = n-1; i >= 1; i -= 1){
        for (int j = 0; j < i; j += 1){
            if (cmp(data[j], data[j+1]) > 0){
                CStudent temp = data[j];
                data[j] = data[j+1];
                data[j+1] = temp;
            }
        }
    }
}

int CompareId(const CStudent &lhs, const CStudent &rhs){
    if(lhs.id < rhs.id) return -1;
    else if(lhs.id == rhs.id) return 0;
    else return 1;
}
int CompareName(const CStudent &lhs, const CStudent &rhs){
    return strcmp(lhs.name, rhs.name);
}
int CompareScore(const CStudent &lhs, const CStudent &rhs){
    if(lhs.score < rhs.score) return -1;
    else if(lhs.score == rhs.score) return 0;
    else return 1;
}



int main()
{
    CStudent data[] = {{1, 90, "David"}, {5, 100, "Allen"}, {3, 92, "Alice"}, {2, 85, "Bob"}, {4, 91, "Cyndi"}};
    const int Size = sizeof(data)/sizeof(data[0]);

    sort(data, Size, CompareId);
    for (const auto &e: data) {cout << e << endl;} cout << endl;
    sort(data, Size, CompareName);
    for (const auto &e: data) {cout << e << endl;} cout << endl;
    sort(data, Size, CompareScore);
    for (const auto &e: data) {cout << e << endl;} cout << endl;
}






