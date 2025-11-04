#include <iostream>
using namespace std;

int main() {
    int val=42;
    cout << "Value: " << val << std::endl;
    int* ptr=&val;
    cout << "Pointer Address: " << ptr << std::endl;
    int deref=*ptr;
    cout << "Dereferenced Value: " << deref << std::endl;
    int size =sizeof(ptr);
    cout << "Pointer Size: " << size << " bytes" << std::endl;
    return 0;
}