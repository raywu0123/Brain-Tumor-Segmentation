def readfile():
    lines = []
    with open('./file_list.txt', 'r') as file:
        lines = file.readlines()
        for line_n, line in enumerate(lines):
            line = line.split('/')
            if len(line) == 7:
                lines[line_n] = line[6]
            else:
                lines[line_n] = line[5]
    return lines


def classify(lines):
    AX_C = 0
    AX = 0
    for i in range(len(lines)):
        if lines[i].startswith('case'):
            if lines[i].find('__Ax') != -1 or lines[i].find('__AX') != -1:
                if lines[i].find('+C') != -1:
                    AX_C = AX_C + 1
                else:
                    AX = AX + 1
        if lines[i].startswith('BRAIN'):
            if lines[i].find('ax') != -1 or lines[i].find('AX') != -1\
               or lines[i].find('MPR') != -1:
                AX = AX + 1
    print(AX, AX_C)


def main():
    lines = readfile()
    classify(lines)


if __name__ == '__main__':
    main()
