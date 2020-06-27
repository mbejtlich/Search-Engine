import re

#
# l = ["ch", '', '', 'e', '', 'e', 'se']
# l = list(filter(None, l))
# print(l)

# A class (a set of rules used to create an object)
class Employee:
    number_of_emps = 0
    raise_amount = 1.04
    # Constructor
    def __init__(self,first,last,pay):
        # Attributes
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'
        Employee.number_of_emps += 1

        # Methods are actions within class
    def fullname(self):
        return '{} {}'.format(self.first,self.last)

    def apply_raise(self):
        self.pay = int(self.pay*self.raise_amount)

    @classmethod
    def from_string(cls,emp_str):
        first,last,pay=emp_str.split('-')
        return cls(first,last,pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True

class Developer(Employee):
    raise_amount = 1.10
    def __init__(self,first,last,pay,prog_lang):
        super().__init__(first,last,pay)
        self.prog_lang=prog_lang

# Instance variables (objects) contain data unique to the instance
emp_1 = Employee('Corey','Schafer',50000)
print(emp_1.pay)
emp_1.apply_raise()
print(emp_1.pay)
# print(dir(emp_1))

# x = 'Matthew-Bejtlich-5000'
#
# # t1,t2,t3=x.split('-')
# # print(t1)
# # print(t2)
# # print(t3)
#
# emp1 = Employee.from_string(x)
# print(emp1.first)


content = 'hello'
content = re.sub(r'^"', '',content)
print(content)

