import os

nMon = [100, 200, 300, 500, 613, 817, 1021, 1225, 1429, 1634, 1838, 2042, 2246, 2450, 2655, 2859, 3063, 3267, 3471, 3675]
for mon in nMon:
    command = "python3 generate_network.py -nmon 2 -bond_length 0.5 -box_length_x 20.0 -box_length_y 10.0 -box_length_z 8.0"
    command = command + " -npeg " + str(mon) + " -o network_"+str(mon)+".atom"
    os.system(command)
