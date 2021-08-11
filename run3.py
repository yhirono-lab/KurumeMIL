import subprocess
import sys

# 7/29時点の全コマンドリスト
command_list = [
    ['-c','subtype','-l','normal'],
    ['-c','subtype','-l','normal','--fc'],
    ['-c','leaf','-l','normal'],
    ['-c','leaf','-l','normal','--fc'],
    ['-c','new_tree','-l','normal'],
    ['-c','new_tree','-l','myinvarse'],
    ['-c','new_tree','-l','LDAM','-C','0.1'],
    ['-c','new_tree','-l','LDAM','-C','0.2'],
    ['-c','new_tree','-l','LDAM','-C','0.3'],
    ['-c','new_tree','-l','LDAM','-C','0.5'],
    ['-c','new_tree','-l','normal','-a'],
    ['-c','new_tree','-l','myinvarse','-a'],
    ['-c','new_tree','-l','LDAM','-C','0.1','-a'],
    ['-c','new_tree','-l','LDAM','-C','0.2','-a'],
    ['-c','new_tree','-l','LDAM','-C','0.3','-a'],
    ['-c','new_tree','-l','LDAM','-C','0.5','-a'],
    ['-c','new_tree','-l','normal','--fc'],
    ['-c','new_tree','-l','myinvarse','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.1','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.2','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.3','--fc'],
    ['-c','new_tree','-l','LDAM','-C','0.5','--fc'],
    ['-c','new_tree','-l','normal','--model','vgg11'],
    ['-c','new_tree','-l','myinvarse','--model','vgg11'],
    ['-c','new_tree','-l','LDAM','-C','0.1','--model','vgg11'],
    ['-c','new_tree','-l','LDAM','-C','0.3','--model','vgg11'],
    ['-c','new_tree','-l','LDAM','-C','0.5','--model','vgg11'],
    ['-c','new_tree','-l','normal','--data','add'],
    ['-c','new_tree','-l','myinvarse','--data','add'],
    ['-c','new_tree','-l','LDAM','-C','0.1','--data','add'],
    ['-c','new_tree','-l','LDAM','-C','0.3','--data','add'],
    ['-c','new_tree','-l','LDAM','-C','0.5','--data','add'],
    ['-c','new_tree','-l','normal','--data','add','--model','vgg11'],
    ['-c','new_tree','-l','myinvarse','--data','add','--model','vgg11'],
    ['-c','new_tree','-l','LDAM','-C','0.1','--data','add','--model','vgg11'],
    ['-c','new_tree','-l','LDAM','-C','0.3','--data','add','--model','vgg11'],
    ['-c','new_tree','-l','LDAM','-C','0.5','--data','add','--model','vgg11'],
]

split_list = [
    ['123','4','5'],
    ['234','5','1']
]

args = sys.argv
if len(args)!=2:
    exit()

# mode_option = [[1,0],[2,3],[5,4],[6,7,8]]
mode_option = [list(range(len(command_list)))]
mode_list = mode_option[int(args[1])]

gpu = int(args[1])

for mode in mode_list:
    # for split in split_list:
    #     command1 = ['python','MIL_train_Single.py']+[split[0], split[1]]+['--depth','1','--leaf','01']
    #     command2 = ['--gpu',f'{gpu}']+command_list[mode]
    #     command = command1 + command2
    #     print(command)
    #     subprocess.run(command)

    #     command1 = ['python','MIL_test.py']+[split[0], split[2]]+['--depth','1','--leaf','01']
    #     command2 = ['--gpu',f'{gpu}']+command_list[mode]
    #     command = command1 + command2
    #     print(command)
    #     subprocess.run(command)
    
    command = ['python','make_log_Graphs.py','--depth','1','--leaf','01']+command_list[mode]
    print(command)
    subprocess.run(command)

    # command = ['python','draw_heatmap.py','--depth','1','--leaf','01']+command_list[mode]
    # print(command)
    # subprocess.run(command)

