#ifndef GROUP_STRING
#define GROUP_STRING
#include <iostream>
#include <string.h>

class String{
    friend std::istream &operator>> (std::istream &is, String &s);
public:
    String();
    String(const String &s);
    String(const char* s);
    ~String();
    size_t size() const;
    const char* c_str() const;
    const char &operator[] (const size_t i) const;
    char &operator[] (const size_t i);
    String &operator+= (const String &s);
    String &operator+= (const char* s);
    String &operator+= (char);
    void clear();
    String &operator= (String s);
    String &operator= (char* s);
    String &operator= (char s);
    String &swap (String &rhs);
    String &swap (char* &rhs);
    void append (const String &s);
    void append (const char* s);
    void append (char s);
private:
    size_t size_ = 0;
    char* str_ = nullptr;
};
bool operator== (const String &lhs, const String &rhs);
bool operator== (const char* lhs, const String &rhs);
bool operator== (const String &lhs, const char* rhs);
bool operator!= (const String& lhs, const String& rhs);
bool operator!= (const char* lhs, const String &rhs);
bool operator!= (const String &lhs, const char* rhs);
bool operator<  (const String& lhs, const String& rhs);
bool operator< (const char* lhs, const String &rhs);
bool operator< (const String &lhs, const char* rhs);
bool operator<= (const String& lhs, const String& rhs);
bool operator<= (const char* lhs, const String &rhs);
bool operator<= (const String &lhs, const char* rhs);
bool operator>  (const String& lhs, const String& rhs);
bool operator> (const char* lhs, const String &rhs);
bool operator> (const String &lhs, const char* rhs);
bool operator>= (const String& lhs, const String& rhs);
bool operator>= (const char* lhs, const String &rhs);
bool operator>= (const String &lhs, const char* rhs);
std::ostream &operator<< (std::ostream &os, const String &s);
String operator+ (const String &lhs, const String &rhs);
String operator+ (const char* lhs, const String &rhs);
String operator+ (const String &lhs, const char* rhs);
String operator+ (const String &lhs, char rhs);
String operator+ (char lhs, const String &rhs);
void swap (String &lhs, String &rhs);
void swap (char* &lhs, String &rhs);
void swap (String &lhs, char* &rhs);
#endif