import subprocess

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','40x','--num_gpu','4','-c','subtype'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','345','1','--depth','1','--leaf','01','--mag','40x','--num_gpu','4','-c','subtype'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','451','2','--depth','1','--leaf','01','--mag','40x','--num_gpu','4','-c','subtype'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','512','3','--depth','1','--leaf','01','--mag','40x','--num_gpu','4','-c','subtype'
]
print(commando)
subprocess.run(commando)