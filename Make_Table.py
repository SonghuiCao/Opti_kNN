import pandas as pd
from prettytable import PrettyTable


if __name__ == '__main__':

    # Create a table
    table = PrettyTable()
    table.field_names = ["component-k", "1", "3", "5"]

    a = 0.92750
    b = 0.00593
    c = "(" + str(a) + ", " + str(b) + ")"
    d = [15]
    d.append(c)
    d.append(c)
    d.append(c)
    table.add_row(d)

    print(c)
    print(table)





