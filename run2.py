import subprocess
import sys

command_list = [
    ['-c','subtype','--fc'],
    ['-c','leaf','--fc'], 
    ['-c','new_tree','--fc','-r'], 
    ['-c','new_tree','-l','myinvarse','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.1','--fc','-r'],
    ['-c','new_tree','-l','LDAM','-C','0.3','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.5','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.2','--fc'],
]

split_list = [
    ['123','4','5'],
    ['234','5','1']
]

args = sys.argv
if len(args)!=2:
    exit()

mode_option = [[0,1],[2,3],[4,5],[6,7]]
mode_option = [[2,3,4,7,5,6,0,1]]
mode_list = mode_option[int(args[1])]

gpu = int(args[1])

for mode in mode_list:
    for split in split_list:
        command1 = ['python','MIL_train.py']+[split[0], split[1]]+['--depth','1','--leaf','01']
        command2 = ['--num_gpu','2']+command_list[mode]
        command = command1 + command2
        print(command)
        subprocess.run(command)

        command1 = ['python','MIL_test.py']+[split[0], split[2]]+['--depth','1','--leaf','01']
        command2 = ['--gpu',f'{gpu}']+command_list[mode]
        command = command1 + command2
        command = [c for c in command if c != '-r']
        print(command)
        subprocess.run(command)
    
    command = ['python','make_log_Graphs.py','--depth','1','--leaf','01']+command_list[mode]
    command = [c for c in command if c != '-r']
    subprocess.run(command)

    command = ['python','draw_heatmap.py','--depth','1','--leaf','01']+command_list[mode]
    command = [c for c in command if c != '-r']
    subprocess.run(command)

