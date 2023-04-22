# 手动创建类的方法示例
def man():
    print('aaa')


def man2(self):
    print('aaa')


@classmethod
def clsmethod(cls):
    print(cls)


Man = type('Man', (), {'man': man, 'clsmethod': clsmethod, 'man2': man2})

if __name__ == '__main__':
    aaa = Man()
    Man.man()
