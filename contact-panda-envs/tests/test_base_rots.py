from contact_panda_envs.envs.cabinet.cabinet_contact import PandaCabinetOneFingerNoSTS6DOF


num_steps = 50
num_back_and_forth = 6

env = PandaCabinetOneFingerNoSTS6DOF()

for bf in range(num_back_and_forth):
    if bf % 2 == 0:
        x_cmd = 0.01
    else:
        x_cmd = -0.01
    for s in range(num_steps):
        env.step([0, 0, 0, x_cmd, 0, 0])
