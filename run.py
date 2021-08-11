import subprocess
import sys

# command_list = [
#     ['-c','subtype','--fc'],
#     ['-c','leaf','--fc'], 
#     ['-c','new_tree','--fc','-r'], 
#     ['-c','new_tree','-l','myinvarse','--fc'],
#     ['-c','new_tree','-l','LDAM','-C','0.1','--fc','-r'],
#     ['-c','new_tree','-l','LDAM','-C','0.3','--fc'],
#     ['-c','new_tree','-l','LDAM','-C','0.5','--fc'],
#     ['-c','new_tree','-l','LDAM','-C','0.2','--fc'],
# ]
# command_list = [
#     ['-c','new_tree','-l','LDAM','-C','0.1','--model','vgg11'],
#     ['-c','new_tree','-l','LDAM','-C','0.3','--model','vgg11'],
#     ['-c','new_tree','-l','LDAM','-C','0.5','--model','vgg11'],
# ]
command_list = [
    # ['-c','new_tree','-l','normal','--model','vgg11'],
    # ['-c','new_tree','-l','normal','--data','add'],
    # ['-c','new_tree','-l','normal','--data','add','--model','vgg11'],
    # ['-c','new_tree','-l','focal','-g','1.0','--data','add','-r'],
    # ['-c','new_tree','-l','focal','-g','1.5','--data','add'],
    # ['-c','new_tree','-l','focal','-g','2.0','--data','add'],
    # ['-c','new_tree','-l','focal','-g','1.0','--data','add','--model','vgg11'],
    # ['-c','new_tree','-l','focal','-g','1.5','--data','add','--model','vgg11'],
    # ['-c','new_tree','-l','focal','-g','2.0','--data','add','--model','vgg11'],
    ['--depth','2','--leaf','01','-c','new_tree','-l','normal','--data','add','-r'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','focal','-g','2.0','--data','add','-r'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','focal','-g','1.0','--data','add','--model','vgg11'],
    ['--depth','1','--leaf','01','-c','new_tree','-l','focal','-g','2.0','--data','add','--model','vgg11'],
    ['--depth','2','--leaf','01','-c','new_tree','-l','myinvarse','--data','add'],
    ['--depth','2','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add'],
    ['--depth','2','--leaf','01','-c','new_tree','-l','normal','--data','add','--model','vgg11'],
    ['--depth','2','--leaf','01','-c','new_tree','-l','myinvarse','--data','add','--model','vgg11'],
    ['--depth','2','--leaf','01','-c','new_tree','-l','LDAM','-C','0.5','--data','add','--model','vgg11'],
]

split_list = [
    ['123','4','5'],
    ['234','5','1']
]

args = sys.argv
if len(args)!=2:
    exit()

# mode_option = [[0,1],[2,3],[4,5],[6,7]]
# mode_list = mode_option[int(args[1])]
mode_list = range(len(command_list))

gpu = int(args[1])

for mode in mode_list:
    for split in split_list:
        command1 = ['python','MIL_train.py']+[split[0], split[1]]
        command2 = ['--num_gpu','4']+command_list[mode]
        command = command1 + command2
        print(command)
        subprocess.run(command)

        command1 = ['python','MIL_test.py']+[split[0], split[2]]
        command2 = ['--gpu',f'{gpu}']+command_list[mode]
        command = command1 + command2
        command = [c for c in command if c != '-r']
        print(command)
        subprocess.run(command)
    
    command = ['python','make_log_Graphs.py']+command_list[mode]
    command = [c for c in command if c != '-r']
    print(command)
    subprocess.run(command)

    command = ['python','draw_heatmap.py']+command_list[mode]
    command = [c for c in command if c != '-r']
    print(command)
    subprocess.run(command)

