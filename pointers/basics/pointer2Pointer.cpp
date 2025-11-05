#include <iostream>
using namespace std;


int main(){
    int val=42;
    int* ptr=&val;
    int** pptr=&ptr;
    cout << "Value: " << val << std::endl;
    cout << "Pointer Address: " << ptr << std::endl;
    cout << "Pointer to Pointer Address: " << pptr << std::endl;

    cout<< " Whether the value of **pptr and val are same: " << (**pptr == val) << std::endl;
    const int deref=**pptr;
    cout << "Dereferenced Value using **pptr: " << deref << std::endl;

    return 0;

}