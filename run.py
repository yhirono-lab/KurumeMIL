import subprocess

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','40x','--num_gpu','2','-c','new_tree','-l','LDAM','-C','0.125'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','123','4','--depth','1','--leaf','01','--mag','40x','--num_gpu','2','-c','new_tree','-l','LDAM','-C','0.175'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','40x','--num_gpu','2','-c','new_tree','-l','LDAM','-C','0.175'
]
print(commando)
subprocess.run(commando)