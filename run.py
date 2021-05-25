import subprocess

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','40x'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','20x'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','10x'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','234','5','--depth','1','--leaf','01','--mag','5x'
]
print(commando)
subprocess.run(commando)


commando = [
    'python','MIL_train.py','345','1','--depth','1','--leaf','01','--mag','40x'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','345','1','--depth','1','--leaf','01','--mag','20x'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','345','1','--depth','1','--leaf','01','--mag','10x'
]
print(commando)
subprocess.run(commando)

commando = [
    'python','MIL_train.py','345','1','--depth','1','--leaf','01','--mag','5x'
]
print(commando)
subprocess.run(commando)
