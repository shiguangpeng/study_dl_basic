class Animal:
    a = 4

    @classmethod
    def run(cls):
        print(cls)


class Person:
    __metaclass__ = Animal

    # @classmethod
    # def run(cls):
    #     print('aaaa')


if __name__ == '__main__':
    p = Person
