
def main():
    lines = []
    with open('./file_list.txt', 'r') as file:
        lines = file.readlines()
        for line_n, line in enumerate(lines):
            lines[line_n] = lines[line_n].split('/')
            if len(lines[line_n]) == 7:
                lines[line_n] = lines[line_n][6]
            else:
                lines[line_n] = lines[line_n][5] 

    AX_C = 0
    AX = 0
    mra = 0
    for i in range(len(lines)):
        if lines[i][0] == 'c':
            lines[i] = lines[i].split('_')
            for j in range(5, len(lines[i])):
                if lines[i][j] == 'AX' or lines[i][j] == 'Ax'\
                   or lines[i][j] == 'MPRAGE' or lines[i][j] == 'MPR':
                    if len(lines[i][8]) > 3:
                        if lines[i][8][2] == '+':
                            AX_C = AX_C + 1                        
                        else:
                            AX = AX + 1
                    else:
                        AX = AX + 1   
                    break  
                elif lines[i][j] == 'TOF':
                    mra = mra + 1
                    break

        if lines[i][0] == 'B': 
            c = 0
            lines[i] = lines[i].split('_')
            for j in range(5, len(lines[i])):
                if lines[i][j] == '+C':
                    c = 1

            for j in range(5, len(lines[i])):
                if lines[i][j] == 'AX' or lines[i][j] == 'ax' \
                   or lines[i][j] == 'MPRAGE' or lines[i][j] == 'MPR'\
                   or lines[i][j] == 'TOF':
                    if c == 1:
                        AX_C = AX_C + 1 
                    else:
                        AX = AX + 1 
                    break 
                elif lines[i][j] == 'TOF':
                    mra = mra + 1
                    break    
        lines[i] = []    
    print(AX, AX_C)   


if __name__ == '__main__':
    main()