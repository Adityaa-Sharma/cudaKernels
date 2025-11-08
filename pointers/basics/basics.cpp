#include <iostream>
using namespace std;

void solve(int arr[], int n) {
    // Function to demonstrate pointer usage
    cout << " Size of the array in function: " << sizeof(arr) << " bytes" << std::endl;
}


int main() {
    int val=42;
    cout << "Value: " << val << std::endl;
    int* ptr=&val;
    cout << "Pointer Address: " << ptr << std::endl;
    int deref=*ptr;
    cout << "Dereferenced Value: " << deref << std::endl;
    int size =sizeof(ptr);
    cout << "Pointer Size: " << size << " bytes" << std::endl;

    int arr[5] = {1, 2, 3, 4, 5};
    cout << " Size of the array in main: " << sizeof(arr) << " bytes" << std::endl;
    solve(arr, 5);
    return 0;
}