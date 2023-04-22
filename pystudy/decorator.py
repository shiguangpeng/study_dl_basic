def getdecorate(check):
    print(check)

    def decorate(func):
        print('outer')

        def inner(name):
            func(name)

        return inner

    return decorate


# @getdecorate('check......')
# def send(name):
#     print('xxx.{0}'.format(name))







if __name__ == '__main__':
    send('okyousgp')
