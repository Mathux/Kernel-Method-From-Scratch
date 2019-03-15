from src.tools.utils import Logger


class Init(type):
    def __new__(cls, name, bases, dct):
        print("meta: creating {} {}".format(name, bases))
        print("dct:", dct)
        return type.__new__(cls, name, bases, dct)
        

class Test1(Logger):
    def __init__(self, args):
        self.args = args

    def test1(self):
        return "1"

    def test(self):
        return self.args + self.test1()


class Test2(Logger):
    def __init__(self, args):
        self.args = args

    def test2(self):
        return "1"

    def test(self):
        return self.args + self.test2()

    
class Test(Logger, metaclass=Init):
    

t = Test(4)

