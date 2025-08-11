#include <iostream>

using namespace std;

int main(){
    int num=6513249; // 0x00636261
    int *pnum=&num;
    char *pstr=reinterpret_cast<char*>(pnum);
    cout << "*pnum value: " << *pnum << endl; // 6513249
    cout << "*pstr value: " << *pstr << endl; // a
    cin.get();
    return 0;
}