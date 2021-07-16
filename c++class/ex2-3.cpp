#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct CStudent
{
    int id,
        score;
    char name[30];
};
void scan(CStudent *s)
{
    scanf("%d %s %d", &s->id, s->name, &s->score);
}
void print(const CStudent *s)
{
    printf("%d %s %d\n", s->id, s->name, s->score);
}
using functionPointer = int(*)(const CStudent &lhs, const CStudent &rhs);
int CompId(const void *l, const void *r){
    const CStudent *lhs = (const CStudent *)l;
    const CStudent *rhs = (const CStudent *)r;
    if(lhs->id < rhs->id) return -1;
    else if(lhs->id == rhs->id) return 0;
    else return 1;
}
int CompName(const void *l, const void *r){
    const CStudent *lhs = (const CStudent *)l;
    const CStudent *rhs = (const CStudent *)r;
    return strcmp(lhs->name, rhs->name);
}

int main()
{
    CStudent students[] = {{1, 90, "David"}, {5, 100, "Allen"}, {3, 92, "Alice"}, {2, 85, "Bob"}, {4, 91, "Cyndi"}};
    const int Size = sizeof(students)/sizeof(students[0]);

    qsort(students, Size, sizeof(CStudent), CompId);
    for (int i=0; i<Size; i+=1)
    {
        print(&students[i]);
    }

    qsort(students, Size, sizeof(CStudent), CompName);
    for (int i=0; i<Size; i+=1)
    {
        print(&students[i]);
    }
    return 0;
}

