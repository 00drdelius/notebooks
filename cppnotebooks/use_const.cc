#include <iostream>

/**
 * `const`关键字用于修饰非指针变量时，`const int`与`int const`等价。
 * - `const int cons1`：
 * 这个`const`修饰的是`cons1`变量，因此`cons1=10`这种操作非法
 * 
 * 然而若是用于修饰指针变量，其修饰所在的位置会造成很大的不同：
 * - `const int *p`：
 * 这个`const`修饰的变量是`*p`。所以`*p=2`这种操作非法；`p=&x`这种操作合法
 * 
 * - `int *const p3=&x`：
 * 这个`const`修饰的变量是`p3`。所以`p3=&y`这种操作非法；`*p3=10`这种操作合法
 * 
 * - `const int *const p4=&x`：
 * 第一个`const`修饰的是`*p4`，第二个`const`修饰的是`p4`，所以`p4=&y`与`*p4=10`都非法
 * 
 * ### 注：
 * - 对于变量的数据类型定义`int`等 其实你放在哪都无所谓，毕竟也是一种对变量的修饰符
 * - `*p`在加上数据类型`int`、关键字`const`修饰后可能看着有点怪，你可以将整体想象成一种特殊变量
 */
void const_variable(){
    const int cons1=2;
    cons1=10;

    int x=2, y=3;
    const int *p; // 指向常量的指针
    int *p2;

    p=&x;
    p=&y;
    *p=2;

    *p2=2;

    int *const p3=&x; // 常量指针
    p3=&y;
    *p3=x;

    const int *const p4=&x;
    *p4=10;
    p4=&x;
}

namespace use_const
{
class Date
{
public:
    Date(int y, int m, int d):
        _year(y),_month(m),_day(d) {}
    
    /**
     * 相当于`void set_year( Date *this )`
     */
    void set_year() {
        this->_year=2023;
    }

    /**
     * 相当于`int get_year_( const Date *this )`
     * 但是类成员函数不可以显式指明 隐含的 this指针，因此cpp标准将 Date 前的 const 放到函数参数后，即下面的函数。
     * 
     * `const`关键字修饰`*this`，因此`(*this).year=10`非法
     */
    int get_year_() const {
        // (*this)._year=10; ilegal
        return this->_year;
    }
    /**
     * 相当于`const int get_year2( const Date *this )`，
     * 返回值不可修改，`*this`也不可修改；
     */
    const int get_year2() const {return this->_year;}
    /**
     * 这里的`const`表示函数返回的是`const int`，返回值不可修改
     */
    const int get_year() {
        const int const_year=_year;
        return const_year;
    }
    //就是`const int get_year()`
    int const get_year() {
        const int const_year=_year;
        return const_year;
    }
private:
    int _year, _month, _day;
};
}

void Printf(const ::use_const::Date& d){
    d.Printf();
    d.Printf2();
    d.Printf3();
    d.Printf4();
}

int main(){
    ::use_const::Date d(2025,4,3);
    d.Printf();
}