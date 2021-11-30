
to_print=[]
for i in range(0,11*27):
    to_print.append(str(i) + ": RIGHT, " )
    if i+1%11==0:
        to_print.append("/n/n")

print("".join(to_print))