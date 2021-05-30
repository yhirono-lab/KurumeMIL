import subprocess

commando = [
    'python','MIL_train.py','123','4','--depth','1','--leaf','01','--mag','40x','--num_gpu','4','-c','new_tree','--loss_mode','invarse'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','40x','--num_gpu','4','-c','new_tree','--loss_mode','invarse'
]
print(commando)
subprocess.run(commando)