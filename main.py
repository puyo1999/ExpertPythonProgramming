# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import RNN.RNN
import RNN.Netflix
import OriginKerasPractice

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    '''
    match sys.platform:
        case "windows":
            print("Running on Windows")
        case _:
            raise NotImplementedError(
                f"{sys.platform} not supported!"
            )
            '''

def fizzbuzz():
    for i in range(100):
        match(i % 3, i % 5):
            case(0,0): print("FizzBuzz")
            case(0,_): print("Fizz")
            case(_,0): print("Buzz")
            case _: print(i)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    #fizzbuzz()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
