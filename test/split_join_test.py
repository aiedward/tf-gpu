"""
@Project   : tf-gpu
@Module    : split_join_test.py
@Author    : Jeff [arfu.guo@gmail.com]
@Created   : 2018/12/31 上午12:20
@Desc      : 
"""


def split_test():
    a = " I live in Shanghai "
    b = a.split(" ")
    print(b)


def convert(original):
    a = " " + original + " "
    print(a + ".")
    b = original
    print(b + ".")


if __name__ == "__main__":
    split_test()
    convert("I live in Shanghai")
