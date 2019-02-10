class Test1(object):
    def __init__(self, temp):
        print(temp, 'is Printed')

class Test2(Test1):
    def __init__(self):
        print('First Init')
        super().__init__('Test')
        
new = Test2()